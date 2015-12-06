__author__ = 'Lukas Woodtli'

from os.path import dirname, join
import evaluator

DATA_DIR = join(dirname(__file__), "..", "data")


args = ["", join(DATA_DIR, "articles.txt"), join(DATA_DIR, "log.txt")]

evaluator.main(args)