from torch import nn
import torch
import numpy as np
from symbolicregression.model.model_wrapper import ModelWrapper
from symbolicregression.model.sklearn_wrapper import SymbolicTransformerRegressor , get_top_k_features
import time 


class Transformer(nn.Module):
    def __init__(self, model, params, env, samples):
        super().__init__()
        self.model = model
        #self.model = torch.load('./symbolicregression/weights/model.pt')
        self.first_dropout = nn.Dropout(0.1)
        self.params = params
        self.env = env
        self.embedder, self.encoder, self.decoder = self.model.embedder, self.model.encoder, self.model.decoder
        self.samples = samples

        x_to_fit = samples['x_to_fit']
        y_to_fit = samples['y_to_fit']
        x1 = []
        for seq_id in range(len(x_to_fit)):
            x1.append([])
            for seq_l in range(len(x_to_fit[seq_id])):
                if np.isscalar(y_to_fit[seq_id][seq_l]):
                    x1[seq_id].append([x_to_fit[seq_id][seq_l], np.array([y_to_fit[seq_id][seq_l]])])
                else:
                    x1[seq_id].append([x_to_fit[seq_id][seq_l], y_to_fit[seq_id][seq_l]])
        self.x1, self.src_len = self.embedder(x1)
        self.encoded = self.encoder("fwd", x=self.x1, lengths=self.src_len, causal=False).transpose(0, 1)

    def generate_beams(self, input_ids,
                top_k,
                num_beams,
                length_penalty,
                early_stopping,
                max_length,
                top_k_hash,
                use_prefix_cache
                ):

        decoded, tgt_len, generated_hyps, top_k_hash = self.decoder.generate_beam_from_state(
            self.encoded, self.src_len, input_ids, num_beams,top_k, top_k_hash, use_prefix_cache,length_penalty, early_stopping, max_len=200)

        return generated_hyps, top_k_hash

    def top_k(self, input_ids, top_k):

        top_k_tokens = self.decoder.extract_top_k(
            self.encoded, self.src_len, input_ids, top_k, max_len=200
        )

        return top_k_tokens

    
def pred_for_sample_no_refine(model, env,preds,x_to_fit):
    mw = ModelWrapper(
            env=env,
            embedder=model.embedder,
            encoder=model.encoder,
            decoder=model.decoder,)
    generations_tree = list(filter(lambda x: x is not None,
                [env.idx_to_infix(preds[1:], is_float=False, str_array=False)]))
    if generations_tree == []: 
        y = None
        model_str= None
    else:
        numexpr_fn = mw.env.simplifier.tree_to_numexpr_fn(generations_tree[0]) 
        y = numexpr_fn(x_to_fit[0])[:,0]  
        model_str = generations_tree[0].infix()
    return y, model_str , generations_tree


def pred_for_sample(model, env,x_to_fit,y_to_fit, refine=True, beam_type='search', beam_size=1):
    mw = ModelWrapper(
            env=env,
            embedder=model.embedder,
            encoder=model.encoder,
            decoder=model.decoder,
            beam_type= beam_type,
            beam_size= beam_size)
    dstr = SymbolicTransformerRegressor(model=mw,n_trees_to_refine=beam_size)
    replace_ops = {"add": "+", "mul": "*", "sub": "-", "pow": "**", "inv": "1/"}
    dstr.fit(x_to_fit[0], y_to_fit[0], verbose=False)
    if refine:
        tree = dstr.retrieve_tree(with_infos=True)["relabed_predicted_tree"]
    else:
        tree = dstr.retrieve_tree(refinement_type="NoRef", with_infos=True)["relabed_predicted_tree"]
    model_str = tree.infix()
    for op,replace_op in replace_ops.items():
        model_str = model_str.replace(op,replace_op)
    numexpr_fn = mw.env.simplifier.tree_to_numexpr_fn(tree) 
    y = numexpr_fn(x_to_fit[0])[:,0]  
    return y, model_str , [tree]


def refine_for_sample(params, model, env, preds ,x_to_fit, y_to_fit):
    embedder = model.embedder
    encoder = model.encoder
    decoder = model.decoder
    embedder.eval()
    encoder.eval()
    decoder.eval()

    mw = ModelWrapper(
            env=env,
            embedder=embedder,
            encoder=encoder,
            decoder=decoder,
            beam_length_penalty=params.beam_length_penalty,
            beam_size=params.beam_size,
            max_generated_output_len=params.max_generated_output_len,
            beam_early_stopping=params.beam_early_stopping,
            beam_temperature=params.beam_temperature,
            beam_type=params.beam_type,
        )
    
    dstr = SymbolicTransformerRegressor(
            model=mw,
            max_input_points=params.max_input_points,
            n_trees_to_refine=params.n_trees_to_refine,
            max_number_bags=params.max_number_bags,
            rescale=params.rescale,
        )

    replace_ops = {"add": "+", "mul": "*", "sub": "-", "pow": "**", "inv": "1/"}

    generations_tree = list(filter(lambda x: x is not None,
                [env.idx_to_infix(preds[1:], is_float=False, str_array=False)]))

    if generations_tree == []: 
        y = None
        model_str= None
        tree = None
    else:
        dstr.start_fit = time.time()
        X = x_to_fit[0]
        Y = y_to_fit[0]
        if not isinstance(X, list):
            X = [X]
            Y = [Y]
        n_datasets = len(X)
        dstr.top_k_features = [None for _ in range(n_datasets)]
        for i in range(n_datasets):
            dstr.top_k_features[i] = get_top_k_features(X[i], Y[i], k=dstr.model.env.params.max_input_dimension)
            X[i] = X[i][:, dstr.top_k_features[i]]
        dstr.tree = {}

        refined_candidate = dstr.refine(X[0], Y[0], generations_tree, verbose=False)
        dstr.tree[0] = refined_candidate

        tree = dstr.retrieve_tree(with_infos=True)["relabed_predicted_tree"]
        model_str = tree.infix()
        for op,replace_op in replace_ops.items():
            model_str = model_str.replace(op,replace_op)

        numexpr_fn = mw.env.simplifier.tree_to_numexpr_fn(tree) 
        y = numexpr_fn(x_to_fit[0])[:,0]  

    return y, model_str, [tree]


def refine_for_sample_test(model, env, preds ,x_to_fit, y_to_fit,x_to_pred):
    mw = ModelWrapper(
            env=env,
            embedder=model.embedder,
            encoder=model.encoder,
            decoder=model.decoder,)
    dstr = SymbolicTransformerRegressor(model=mw,)
    replace_ops = {"add": "+", "mul": "*", "sub": "-", "pow": "**", "inv": "1/"}
    generations_tree = list(filter(lambda x: x is not None,
                [env.idx_to_infix(preds[1:], is_float=False, str_array=False)]))
    if generations_tree == []: 
        y = None
        model_str= None
        tree = None
    else:
        dstr.start_fit = time.time()
        X = x_to_fit[0]
        Y = y_to_fit[0]
        if not isinstance(X, list):
            X = [X]
            Y = [Y]
        n_datasets = len(X)
        dstr.top_k_features = [None for _ in range(n_datasets)]
        for i in range(n_datasets):
            dstr.top_k_features[i] = get_top_k_features(X[i], Y[i], k=dstr.model.env.params.max_input_dimension)
            X[i] = X[i][:, dstr.top_k_features[i]]
        dstr.tree = {}

        refined_candidate = dstr.refine(X[0], Y[0], generations_tree, verbose=False)
        dstr.tree[0] = refined_candidate

        tree = dstr.retrieve_tree(with_infos=True)["relabed_predicted_tree"]
        model_str = tree.infix()
        for op,replace_op in replace_ops.items():
            model_str = model_str.replace(op,replace_op)

        numexpr_fn = mw.env.simplifier.tree_to_numexpr_fn(tree) 
        y_train = numexpr_fn(x_to_fit[0])[:,0]  
        y_test = numexpr_fn(x_to_pred[0])[:,0] 

    return y_train, y_test, model_str, [tree]


def respond_to_batch(model, max_target_length=200, top_p=1.0, sample_temperature=None):
    
    generations, gen_len = model.decoder.generate(
                model.encoded,
                model.src_len,
                sample_temperature=None,
                max_len=max_target_length,)
    generations = generations.transpose(0,1)
    return generations, gen_len


def pred_for_sample_test(model, env,x_to_fit,y_to_fit, x_to_pred, refine=True, beam_type='search', beam_size=1):
    mw = ModelWrapper(
            env=env,
            embedder=model.embedder,
            encoder=model.encoder,
            decoder=model.decoder,
            beam_type= beam_type,
            beam_size= beam_size)
    dstr = SymbolicTransformerRegressor(model=mw,)

    replace_ops = {"add": "+", "mul": "*", "sub": "-", "pow": "**", "inv": "1/"}

    dstr.fit(x_to_fit[0], y_to_fit[0], verbose=False)
    if refine:
        tree = dstr.retrieve_tree(with_infos=True)["relabed_predicted_tree"]
    else:
        tree = dstr.retrieve_tree(refinement_type="NoRef", with_infos=True)["relabed_predicted_tree"]
    model_str = tree.infix()
    for op,replace_op in replace_ops.items():
        model_str = model_str.replace(op,replace_op)

    numexpr_fn = mw.env.simplifier.tree_to_numexpr_fn(tree) 
    y_train = numexpr_fn(x_to_fit[0])[:,0] 
    y_test = numexpr_fn(x_to_pred[0])[:,0] 

    return y_train, y_test, model_str , [tree]
