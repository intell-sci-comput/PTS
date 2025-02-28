import operator
import random
from typing import List

from deap import gp, algorithms
from deap import tools
from deap import creator
from deap import base

from ..config import Config
from ..expr_utils.exp_tree_node import Expression
from .agent import Agent as Game_GA
# import utils as utils
from .utils import multi_mutate, deap_to_tokens, tokens_to_deap

run = False


class GAPipeline:
    def __init__(self, config_s: Config):
        """
        Initializing the genetic algorithm
        """
        self.exp_dict = config_s.exp_dict
        toolbox = base.Toolbox()
        self.config_s = config_s
        self.agent = Game_GA(toolbox=toolbox, config_s=config_s)
        var_num = sum([exp.child == 0 for num, exp in self.exp_dict.items()])
        pset = gp.PrimitiveSet("MAIN", var_num)
        var_count = 0
        """
        Initializing the tokens of genetic algorithm
        """
        for num, exp in self.exp_dict.items():
            if not isinstance(exp, Expression): continue
            if exp.child == 0:
                pset.renameArguments(**{f'ARG{var_count}': f"exp{num}"})
                var_count += 1
            else:
                pset.addPrimitive(exp.func, exp.child, name=f"exp{num}")
        """
        Initializing the toolbox of genetic algorithm
        """
        global run
        if not run:
            run = True
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
            creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)
        toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("compile", gp.compile, pset=pset)
        toolbox.register("evaluate", self.agent.fitness)
        toolbox.register("select", tools.selTournament, tournsize=config_s.gp.tournsize)
        toolbox.register("mate", gp.cxOnePoint)
        toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
        toolbox.register("mutate", multi_mutate, expr=toolbox.expr_mut, pset=pset)
        toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=config_s.gp.max_height))
        toolbox.decorate("mutate",
                         gp.staticLimit(key=operator.attrgetter("height"), max_value=config_s.gp.max_height))
        self.toolbox = toolbox
        self.pset = pset

    def ga_play(self, pop_init: List[List[int]]) -> List[List[int]]:
        """
        Genetic Algorithm Run Function
        :param pop_init: part of the initial population
        :return: Result of the genetic algorithm(list of token list)
        """
        hof = tools.HallOfFame(20)
        pops = pop_init
        pop = self.config_s.gp.pops
        if len(pops) >= pop // 2: pops = random.sample(pops, pop // 2)
        pops = [creator.Individual(tokens_to_deap(p, self.pset)) for p in pops]
        pops += self.toolbox.population(n=pop - len(pops))
        _ = algorithms.eaSimple(pops, self.toolbox, self.config_s.gp.cxpb, self.config_s.gp.mutpb,
                                self.config_s.gp.times, halloffame=hof, verbose=False)
        return [deap_to_tokens(tokens) for tokens in hof]
