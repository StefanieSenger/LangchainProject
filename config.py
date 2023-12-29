import os
import sys


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
common_root = os.path.join(project_root, 'common_root')
sys.path.insert(0, common_root)
