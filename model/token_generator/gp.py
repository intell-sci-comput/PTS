from .base import TokenGenerator
import re
import numpy as np
import random
import yaml
import json
import itertools
from .GP.model.config import Config
from .GP.model.pipeline import Pipeline
import sympy as sp
import random
import itertools
from collections import Counter
from utils.exprutils import has_nested_func
MAX_LEN_SET = 1000
SAMPLE_PROB = 0.5
SAMPLE_PROB_CROSS_VAR = 0.5
MAX_INTEGER = 10


def read_yaml_to_json(file_path):
    with open(file_path, "r") as file:
        try:
            config = yaml.safe_load(file)
            return json.dumps(config)
        except yaml.YAMLError as e:
            print(f"Error reading YAML file: {e}")
            return None

def get_max_depth(expr):
    def traverse(subexpr, current_depth):
        max_depth = current_depth
        for arg in subexpr.args:
            max_depth = max(max_depth, traverse(arg, current_depth + 1))
        return max_depth

    return traverse(expr, 0)

def get_subexpressions_at_depth(expr, depth):
    if depth == 0:
        return [expr]

    subexprs = []

    def traverse(subexpr, current_depth):
        if current_depth == depth:
            subexprs.append(subexpr)
        else:
            for arg in subexpr.args:
                traverse(arg, current_depth + 1)

    traverse(expr, 0)
    return subexprs

def get_last_subexprs(expr, depth=4):
    max_depth = get_max_depth(expr)
    ret = []
    for d in range(0, max_depth + 1):
        subexprs = get_subexpressions_at_depth(expr, d)
        if max_depth - d < depth:
            ret.extend(subexprs)
    ret = list(set(ret))
    ret.sort(key=lambda x: x.count_ops())
    return ret

def has_large_integer(expr):
    if isinstance(expr, str):
        expr = sp.S(expr)
    for atom in expr.atoms():
        if isinstance(atom, sp.Integer) and abs(int(atom)) > MAX_INTEGER:
            return True
        if isinstance(atom, sp.Rational) and (
            abs(atom.p) > MAX_INTEGER or abs(atom.q) > MAX_INTEGER
        ):
            return True

    return False

def generate_cross_variable(variables, n_sample):
    operations = ["*", "+", "/"]
    all_combinations = list(itertools.combinations_with_replacement(variables, 2))
    if len(all_combinations) < n_sample:
        n_sample = len(all_combinations)

    sampled_combinations = random.sample(all_combinations, n_sample)
    cross_variables = []

    for var1, var2 in sampled_combinations:
        op = random.choice(operations)
        if var1 == var2:
            if op == "/":
                cross_variables.append(f"{var1}")
            else:
                cross_variables.append(f"{var1}{op}{var1}")
        elif op == "/" and random.random() < 0.5:
            cross_variables.append(f"{var2}{op}{var1}")
        else:
            cross_variables.append(f"{var1}{op}{var2}")
    return cross_variables

class GP_TokenGenerator(TokenGenerator):
    def __init__(
        self,
        regressor,
        config,
        variables,
        operators_op,
        use_const,
        n_inputs,
        use_extra_const=False,
    ):

        self.config = config
        self.variables = variables
        self.operators = operators_op
        self.use_const = use_const
        self.n_inputs = n_inputs
        self.token = []
        self.token_generator_model = None
        self.visited_set = set()

        self.trying_const_range = regressor.trying_const_range

        self.CONST_LB = self.trying_const_range[0]
        self.CONST_UB = self.trying_const_range[1]

        self.use_extra_const = use_extra_const
        self.EXTRA_CONST = ["pi"]

    def sample_const(self, use_float_const):
        if use_float_const:
            return round(random.uniform(self.CONST_LB, self.CONST_UB), 1)
        else:
            CONST_LIST = np.linspace(
                self.CONST_LB, self.CONST_UB, self.CONST_UB - self.CONST_LB + 1
            ).tolist()
            CONST_LIST = [x for x in CONST_LIST if x != 0]
            if self.use_extra_const:
                CONST_LIST.extend(self.EXTRA_CONST)
            return random.choice(CONST_LIST)

    def step(
        self,
        n_psrn_tokens,
        n_sample_variables,
        X,
        y,
        use_set=True,
        reset=False,
        use_float_const=False,
    ):
        """
        Generate tokens for symbolic regression.
        
        Args:
            n_psrn_tokens: Total number of tokens to generate
            n_sample_variables: Number of variable tokens
            X: Input features matrix
            y: Target values
            use_set: Whether to ensure unique token sets
            reset: Whether to reset the token generator model
            use_float_const: Whether to use floating point constants
        
        Returns:
            Tuple of (best_expression, sampled_tokens)
        """
        # Calculate token count and prepare data
        n_tokens = n_psrn_tokens - n_sample_variables
        y = y.reshape(-1)
        X = np.transpose(X, (1, 0))
        
        # Get expressions from model
        best_expr, all_expr_form = self.token_generator_fit_x_y(X, y, self.config, reset=reset)
        best_expr = self.replace_varname([best_expr])[0]
        
        # Process expressions to tokens
        symbols = self.process_all_form_to_tokens(all_expr_form, use_float_const)
        
        # Convert to sympy symbols and gather variations
        symbols_sympy = [sp.S(str(sym)) for sym in symbols]
        symbols_sympy += [e.expand() for e in symbols_sympy] + \
                        [e.together() for e in symbols_sympy] + \
                        [e.powsimp() for e in symbols_sympy] + \
                        [e.radsimp() for e in symbols_sympy]

        # Extract useful subexpressions
        tokens = []
        for expr in symbols_sympy:
            for subexpr in get_last_subexprs(expr):
                if subexpr.count_ops() < 10:  # Only keep manageable expressions
                    tokens.append(subexpr)
        
        # Get frequencies
        token_counts = Counter(tokens)
        all_tokens = list(token_counts.keys())
        frequencies = list(token_counts.values())
        tokens_freq = list(zip(all_tokens, frequencies))

        # Try to find valid token set
        for _ in range(MAX_LEN_SET):
            sampled_tokens = self._generate_token_sample(
                all_tokens, frequencies, tokens_freq, n_tokens, use_float_const
            )
            
            # Convert to strings
            sampled_tokens = [str(t) for t in sampled_tokens]
            
            # If not using set constraints, we're done
            if not use_set:
                return best_expr, sampled_tokens
            
            # Check for variable overlap and uniqueness
            sampled_set = set(sampled_tokens)
            if len(sampled_set) != len(set(sampled_tokens + self.variables)) - len(self.variables):
                continue
                
            # Check if this set is new
            set_key = str(sampled_set)
            if set_key not in self.visited_set:
                self.visited_set.add(set_key)
                return best_expr, sampled_tokens
        
        # Return last attempted sample if we couldn't find a valid one
        return best_expr, sampled_tokens

    def _generate_token_sample(self, all_tokens, frequencies, tokens_freq, n_tokens, use_float_const):
        """Generate a sample of tokens based on available pool and constraints."""
        # If we have enough tokens to sample from
        if len(all_tokens) > n_tokens:
            sampled_tokens = []
            
            # Try to collect unique valid tokens
            for _ in range(MAX_LEN_SET):
                if len(sampled_tokens) >= n_tokens:
                    break
                    
                # Choose sampling strategy
                if random.random() < SAMPLE_PROB:
                    if random.random() < SAMPLE_PROB_CROSS_VAR:
                        # Sample from cross-variables
                        chosen_token = sp.S(generate_cross_variable(self.variables, 1)[0])
                    else:
                        # Sample a constant
                        chosen_token = sp.S(self.sample_const(use_float_const))
                else:
                    # Sample from token pool based on frequency
                    chosen_token = random.choices(
                        [token for token, _ in tokens_freq],
                        weights=[freq for _, freq in tokens_freq],
                        k=1
                    )[0]
                
                # Validate the token
                if chosen_token is None:
                    continue
                    
                if (not (not use_float_const and "." in str(chosen_token))
                    and str(chosen_token) not in self.variables
                    and not has_nested_func(chosen_token)
                    and not has_large_integer(chosen_token)
                    and chosen_token not in sampled_tokens):
                    sampled_tokens.append(chosen_token)
            
            # If we couldn't get enough tokens, use random sampling
            if len(sampled_tokens) < n_tokens:
                sampled_tokens = random.choices(all_tokens, weights=frequencies, k=n_tokens)
        else:
            # Not enough tokens, supplement with constants
            sampled_constants_num = n_tokens - len(all_tokens)
            sampled_constants = [self.sample_const(use_float_const) for _ in range(sampled_constants_num)]
            sampled_tokens = all_tokens + sampled_constants
        
        return sampled_tokens

    def process_all_form_to_tokens(self, all_expr_form, use_float_const):
        new_expr_forms = []
        for expr in all_expr_form:
            if len(expr) == 0:
                continue
            elif expr.endswith("+"):
                expr = expr[:-1]
            elif expr.endswith("**"):
                expr = expr[:-2]
            elif expr.endswith("*"):
                expr = expr[:-1]
            expr = expr.replace("C", str(self.sample_const(use_float_const)))
            new_expr_forms.append(expr)

        new_expr_forms = list(set(new_expr_forms))
        new_expr_forms = self.replace_varname(new_expr_forms)
        return new_expr_forms

    def replace_varname(self, new_expr_forms):
        for i in range(len(new_expr_forms)):
            for j in range(len(self.variables)):
                new_expr_forms[i] = new_expr_forms[i].replace(
                    "X" + str(j + 1), self.variables[j]
                )
        return new_expr_forms

    def reward(self, reward, expressions):
        self.token_generator_model.use_psrn_reward_expressions_to_update(expressions)

    def token_generator_fit_x_y(self, x, y, gp_config, reset=False):

        if self.token_generator_model is None or reset:
            config = Config()
            config.json(gp_config)
            config.set_input(x=x, t=y, x_=x, t_=y, tokens=gp_config["base"]["tokens"])
            print("=" * 40)
            print("GP config:", config)
            print("=" * 40)
            self.token_generator_model = Pipeline(config=config)
            clear = True
        else:
            clear = False

        best_exprs, exprs = self.token_generator_model.fit(clear=clear)
        exprs = [best_exprs] + exprs
        return best_exprs, exprs
