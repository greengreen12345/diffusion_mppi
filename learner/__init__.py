#from .normal import NormalLearner
from .hgg import HGGLearner

learner_collection = {

	'hgg': HGGLearner,
}

def create_learner(args):
	return learner_collection[args.learn](args)