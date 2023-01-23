import os
import dill             # version 0.3.6 (store python object into file)
import pkg_resources
import copy
from utils import RequestLimiter


# Naming convention
IS_DEBUG = os.getenv("DEBUG", True)                     # os.getnv() return the value of the environment key if it exists otherwise return the default valu. # NOQA : E501(for line too long)
APP_TITLE = " Classification model"
DESCRIPTION = " A Simple Api that use NLP model to predict Cybersecurity_Text or Not_Cybersecurity_Text "
APP_VERSION = '0.82.01'

MAX_SEQUENCE_LENGTH = 19
CONCURRENT_REQUEST_PER_WORKER: int = 19
# CONCURRENT_REQUEST_PER_WORKER = int(os.getenv("CONCURRENT_REQUEST_PER_WORKER", 20))


# Load the Model
# File_name = os.getenv("lstm.h5", "not found")
# Model_paths = os.getenv("/home/rahul/Downloads/NLP/lstm/")
# Model_file_paths = os.path.join(Model_paths, File_name)
#
#
# def load_model():
#     with open(Model_file_paths, 'rb') as Model_file:
#         return dill.load(Model_file)  # return file
#
# Loaded_model = load_model()




def decode_text(score):
  return "cybersecurity_text" if score > 0.5 else "Not_cyber_security"





