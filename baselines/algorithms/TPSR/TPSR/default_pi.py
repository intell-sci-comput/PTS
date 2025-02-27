import time

import torch
import numpy as np
from value_func import ValueFunc

class E2EHeuristic:
    def __init__(self,
                equation_env,
                rl_env,
                model,
                k,
                num_beams,
                horizon,
                device,
                use_seq_cache,
                use_prefix_cache,
                length_penalty,
                train_value_mode=False,
                value_func=None,
                debug=False):
        self.model = model
        self.rl_env = rl_env
        self.equation_env = equation_env
        self.k = k
        self.num_beams = num_beams
        self.horizon = horizon
        self.device = device
        self.length_penalty = length_penalty 
        self.debug = debug
        self.use_seq_cache = use_seq_cache
        self.use_prefix_cache = use_prefix_cache

        self.train_value_mode = train_value_mode

        if self.train_value_mode:
            # fixme hardcoded state dimension
            self.value_func = ValueFunc(state_size=1600, device=self.device)
            if self.use_seq_cache:
                self.use_seq_cache= False
                print("need to turn off use_seq_cache, otherwise some training data are not collected.")
        if value_func is not None:
            self.value_func = value_func
            self.use_value_mode = True
        else:
            self.use_value_mode = False

        self.output_hash = []
        self.top_k_hash = {}
        self.sample_times = 0
        self.candidate_programs = []
        self.terminal_token = self.equation_env.equation_word2id['<EOS>']

    @property
    def is_train_value_mode(self):
        return self.train_value_mode

    @property
    def is_use_value_mode(self):
        return self.use_value_mode

    def get_predict_sequence(self, state, ret_states=False):
        """
        Args:
            ret_states: Return the hidden states of the Transformer in the generation process.
            Only used to train a value function so far.
        Returns:
            Get the most likely sequence starting from state.
        """
        with torch.no_grad():
            encoded_ids = state
            input_ids = torch.LongTensor(encoded_ids).unsqueeze(0).to(self.device)

            if self.use_seq_cache and self.num_beams == 1:
                # If no beam search is used, if the prefix of a previously generated sequences generated state matches
                # state, Transformer will generate the exact sequence. So use cache.
                for cached_ids in self.output_hash:
                    if encoded_ids == cached_ids[:len(encoded_ids)]:
                        if self.debug: print('sequence cache hit')
                        return cached_ids

            start_time = time.time()
            generated_hyps, top_k_hash_updated = self.model.generate_beams(
                input_ids,
                top_k=self.k,
                num_beams=self.num_beams,
                length_penalty = self.length_penalty,
                early_stopping=True,
                max_length=self.horizon,
                top_k_hash = self.top_k_hash,
                use_prefix_cache = self.use_prefix_cache
            )
            self.top_k_hash = top_k_hash_updated
            
            output_ids_list = []
            for b in range(self.num_beams):
                output_ids_list.append(generated_hyps[0].hyp[b][1])

            if len(output_ids_list) > 1:
                # if got multiple output_ids using beam search, pick the one that has the highest reward
                cand_rewards = [self.rl_env.get_reward(output_ids) for output_ids in output_ids_list]
                output_ids = output_ids_list[np.argmax(cand_rewards)]
            else:
                output_ids = output_ids_list[0]

            if self.use_seq_cache:
                self.output_hash.append(output_ids.tolist())

            self.sample_times += 1

            self.candidate_programs.append(output_ids)

            if self.train_value_mode and ret_states:
                return output_ids, last_layers
            else:
                return output_ids

    def get_top_k_predict(self, state):
        """
        Returns:
            A list of k most likely tokens generate in state (descending in their scores)
        """
        with torch.no_grad():
            if self.use_prefix_cache:
                if tuple(state) in self.top_k_hash:
                    if self.debug: print('top-k cache hit')
                    return self.top_k_hash[tuple(state)]

            encoded_ids = state
            input_ids = torch.LongTensor(encoded_ids).unsqueeze(0).to(self.device)

            start_time = time.time()
            
            top_k_tokens = self.model.top_k(input_ids,top_k = self.k)

            top_k_tokens = top_k_tokens.tolist()[0]

            if self.use_prefix_cache:
                self.top_k_hash[tuple(state)] = top_k_tokens

            return top_k_tokens

    def train_value_func(self, states, value):
        self.value_func.train(states, value)

    def update_cache(self, new_state):
        if self.use_seq_cache:
            # clear hashed sequences that are not consistent with new_state
            self.output_hash = list(filter(lambda x: new_state == x[:len(new_state)], self.output_hash))

        if self.use_prefix_cache:
            new_state = tuple(new_state)
            keys_to_remove = []
            for cached_key in self.top_k_hash:
                if cached_key[:len(new_state)] != new_state:
                    keys_to_remove.append(cached_key)
            for k in keys_to_remove: del self.top_k_hash[k]



class NesymresHeuristic:
    def __init__(self,
                rl_env,
                model,
                k,
                num_beams,
                horizon,
                device,
                use_seq_cache,
                use_prefix_cache,
                length_penalty,
                cfg_params,
                train_value_mode=False,
                value_func=None,
                debug=False):
        self.model = model
        self.rl_env = rl_env
        self.k = k
        self.num_beams = num_beams
        self.horizon = horizon
        self.device = device
        self.length_penalty = length_penalty 
        self.cfg_params = cfg_params

        self.use_seq_cache = use_seq_cache
        self.use_prefix_cache = use_prefix_cache

        self.train_value_mode = train_value_mode
        self.debug = debug

        if self.train_value_mode:
            # fixme hardcoded state dimension
            self.value_func = ValueFunc(state_size=1600, device=self.device)
            if self.use_seq_cache:
                self.use_seq_cache= False
                print("need to turn off use_seq_cache, otherwise some training data are not collected.")
        if value_func is not None:
            self.value_func = value_func
            self.use_value_mode = True
        else:
            self.use_value_mode = False

        self.output_hash = []
        self.sample_times = 0
        self.candidate_programs = []
        self.terminal_token = cfg_params.word2id["F"]

    @property
    def is_train_value_mode(self):
        return self.train_value_mode

    @property
    def is_use_value_mode(self):
        return self.use_value_mode

    def get_predict_sequence(self, state, ret_states=False):
        """
        Args:
            ret_states: Return the hidden states of the Transformer in the generation process.
            Only used to train a value function so far.
        Returns:
            Get the most likely sequence starting from state.
        """
        with torch.no_grad():
            encoded_ids = state
            input_ids = torch.LongTensor(encoded_ids).unsqueeze(0).to(self.device)

            if self.use_seq_cache and self.num_beams == 1:
                # If no beam search is used, if the prefix of a previously generated sequences generated state matches
                # state, Transformer will generate the exact sequence. So use cache.
                for cached_ids in self.output_hash:
                    if encoded_ids == cached_ids[:len(encoded_ids)]:
                        if self.debug: print('sequence cache hit')
                        return cached_ids

            start_time = time.time()
   
            generated_hyps = self.model.generate_beam_from_state(
                input_ids,
                self.num_beams,
                self.cfg_params
            )

             # print('generate sequence time: ' + str(time.time() - start_time))
            
            output_ids_list = []
            for b in range(self.num_beams):
                output_ids_list.append(generated_hyps.hyp[b][1])
            
            if len(output_ids_list) > 1:
                # if got multiple output_ids using beam search, pick the one that has the highest reward
                cand_rewards = [self.rl_env.get_reward(output_ids) for output_ids in output_ids_list]
                output_ids = output_ids_list[np.argmax(cand_rewards)]
            else:
                output_ids = output_ids_list[0]

            if self.use_seq_cache:
                self.output_hash.append(output_ids.tolist())
            # breakpoint()

            self.sample_times += 1

            self.candidate_programs.append(output_ids)

            if self.train_value_mode and ret_states:
                return output_ids, last_layers
            else:
                return output_ids
           

    def get_top_k_predict(self, state):
        """
        Returns:
            A list of k most likely tokens generate in state (descending in their scores)
        """
        with torch.no_grad():
            if self.use_prefix_cache:
                top_k_tokens = self.model.top_k_hash.get(state)
                if top_k_tokens is not None:
                    if self.debug: print('top-k cache hit')
                    return top_k_tokens

            encoded_ids = state
            input_ids = torch.LongTensor(encoded_ids).unsqueeze(0).to(self.device)

            start_time = time.time()

            top_k_tokens = self.model.extract_top_k(input_ids,top_k = self.k)

            top_k_tokens = top_k_tokens.tolist()[0]

            return top_k_tokens

    def train_value_func(self, states, value):
        self.value_func.train(states, value)

    def update_cache(self, new_state):
        if self.use_seq_cache:
            # clear hashed sequences that are not consistent with new_state
            self.output_hash = list(filter(lambda x: new_state == x[:len(new_state)], self.output_hash))

        if self.use_prefix_cache:
            # clear hashed key, value pairs that are not consistent with new_state
            self.model.prefix_key_values.clear(new_state)
            self.model.top_k_hash.clear(new_state)