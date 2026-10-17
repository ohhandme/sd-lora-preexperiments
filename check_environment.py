from spec_research.common import *
from spec_research.environment import check_environment
if __name__=='__main__':
    assert_spec()
    with gpu_lock():check_environment()
