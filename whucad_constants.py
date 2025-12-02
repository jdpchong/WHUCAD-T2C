# # 静态参数
# # whucad_constants.py
# """
# Constants and configuration for WHUCAD-style command sequences.
# """
#
# import numpy as np
#
# # Command types from your WHUCAD config
# ALL_COMMANDS = [
#     'Line', 'Arc', 'Circle', 'Spline', 'SCP',
#     'EOS', 'SOL',
#     'Ext', 'Rev', 'Pocket', 'Groove',
#     'Shell', 'Chamfer', 'Fillet', 'Draft', 'Mirror', 'Hole',
#     'Topo', 'Select', 'MirrorStart',
#     'NoSharedIncluded', 'NoSharedIncludedEnd',
#     'AllOrientedIncluded1', 'AllOrientedIncluded2', 'AllOrientedIncludedEnd',
#     'AllPartiallySharedIncluded', 'AllPartiallySharedIncludedEnd'
# ]
#
# N_CMD_TYPES = len(ALL_COMMANDS)  # 27
# N_PARAMS = 32                    # 32 continuous parameter slots per step (vec[:,1:])
#
# # Sequence length setting (WHUCAD paper uses up to 100 steps)
# MAX_SEQ_LEN = 100
#
# # PAD value in WHUCAD vectors (from your config)
# PAD_VAL = -1.0
#
# # You can tweak model-related defaults here if you like
# DEFAULT_D_MODEL = 512
# DEFAULT_N_HEAD = 8
# DEFAULT_NUM_ENCODER_LAYERS = 4
# DEFAULT_NUM_DECODER_LAYERS = 4
# DEFAULT_DIM_FEEDFORWARD = 1024
# DEFAULT_DROPOUT = 0.1
# DEFAULT_MAX_TEXT_LEN = 256




# whucad_constants.py
import numpy as np

ALL_COMMANDS = ['Line', 'Arc', 'Circle', 'Spline', 'SCP', 'EOS', 'SOL', 'Ext',
                'Rev', 'Pocket', 'Groove', 'Shell', 'Chamfer', 'Fillet',
                'Draft', 'Mirror', 'Hole', 'Topo', 'Select', 'MirrorStart',
                'NoSharedIncluded', 'NoSharedIncludedEnd', 'AllOrientedIncluded1',
                'AllOrientedIncluded2', 'AllOrientedIncludedEnd',
                'AllPartiallySharedIncluded', 'AllPartiallySharedIncludedEnd']

LINE_IDX = ALL_COMMANDS.index('Line')
ARC_IDX = ALL_COMMANDS.index('Arc')
CIRCLE_IDX = ALL_COMMANDS.index('Circle')
SPLINE_IDX = ALL_COMMANDS.index('Spline')
SCP_IDX = ALL_COMMANDS.index('SCP')
EOS_IDX = ALL_COMMANDS.index('EOS')
SOL_IDX = ALL_COMMANDS.index('SOL')
EXT_IDX = ALL_COMMANDS.index('Ext')
REV_IDX = ALL_COMMANDS.index('Rev')
POCKET_IDX = ALL_COMMANDS.index('Pocket')
GROOVE_IDX = ALL_COMMANDS.index('Groove')
SHELL_IDX = ALL_COMMANDS.index('Shell')
CHAMFER_IDX = ALL_COMMANDS.index('Chamfer')
FILLET_IDX = ALL_COMMANDS.index('Fillet')
DRAFT_IDX = ALL_COMMANDS.index('Draft')
MIRROR_IDX = ALL_COMMANDS.index('Mirror')
HOLE_IDX = ALL_COMMANDS.index('Hole')
TOPO_IDX = ALL_COMMANDS.index('Topo')
SELECT_IDX = ALL_COMMANDS.index('Select')
MIRROR_START_IDX = ALL_COMMANDS.index('MirrorStart')
NO_SHARED_INCLUDED_IDX = ALL_COMMANDS.index('NoSharedIncluded')
NO_SHARED_INCLUDED_END_IDX = ALL_COMMANDS.index('NoSharedIncludedEnd')
ALL_ORIENTED_INCLUDED_1_IDX = ALL_COMMANDS.index('AllOrientedIncluded1')
ALL_ORIENTED_INCLUDED_2_IDX = ALL_COMMANDS.index('AllOrientedIncluded2')
ALL_ORIENTED_INCLUDED_END_IDX = ALL_COMMANDS.index('AllOrientedIncludedEnd')
ALL_PARTIALLY_INCLUDED_IDX = ALL_COMMANDS.index('AllPartiallySharedIncluded')
ALL_PARTIALLY_INCLUDED_END_IDX = ALL_COMMANDS.index('AllPartiallySharedIncludedEnd')

# ---- 参数数量定义 ----
PAD_VAL = -1

N_ARGS_SKETCH = 5      # sketch parameters: x, y, alpha, f, r
N_ARGS_PLANE = 3       # sketch plane orientation: theta, phi, gamma
N_ARGS_TRANS = 4       # sketch plane origin + sketch bbox size: p_x, p_y, p_z, s
N_ARGS_BODY_PARAM = 7  # length1, length2, length1_type, length2_type, angle1, angle2, boolean
N_ARGS_FINISH_PARAM = 9
N_ARGS_SELECT_PARAM = 4

# ⚠️ 这里要包含 BODY_PARAM
N_ARGS_EXT = N_ARGS_PLANE + N_ARGS_TRANS + N_ARGS_BODY_PARAM

# 整个向量里参数总数（不含第 0 维的 cmd）
N_ARGS = N_ARGS_SKETCH + N_ARGS_EXT + N_ARGS_FINISH_PARAM + N_ARGS_SELECT_PARAM


# 这里只保留你真正需要的 CMD_ARGS_MASK 和 ARGS_N（=256）
CMD_ARGS_MASK = np.array([
    [1, 1, 0, 0, 0, *[0]*N_ARGS_EXT, *[0]*N_ARGS_FINISH_PARAM, *[0]*N_ARGS_SELECT_PARAM],   # Line
    [1, 1, 1, 1, 0, *[0]*N_ARGS_EXT, *[0]*N_ARGS_FINISH_PARAM, *[0]*N_ARGS_SELECT_PARAM],   # Arc
    [1, 1, 0, 0, 1, *[0]*N_ARGS_EXT, *[0]*N_ARGS_FINISH_PARAM, *[0]*N_ARGS_SELECT_PARAM],   # Circle
    [0, 0, 0, 0, 0, *[0]*N_ARGS_EXT, *[0]*N_ARGS_FINISH_PARAM, *[0]*N_ARGS_SELECT_PARAM],   # Spline
    [1, 1, 0, 0, 0, *[0]*N_ARGS_EXT, *[0]*N_ARGS_FINISH_PARAM, *[0]*N_ARGS_SELECT_PARAM],   # SCP
    [*[0]*N_ARGS_SKETCH, *[0]*N_ARGS_EXT, *[0]*N_ARGS_FINISH_PARAM, *[0]*N_ARGS_SELECT_PARAM],  # EOS
    [*[0]*N_ARGS_SKETCH, *[0]*N_ARGS_EXT, *[0]*N_ARGS_FINISH_PARAM, *[0]*N_ARGS_SELECT_PARAM],  # SOL
    [*[0]*N_ARGS_SKETCH, *[1]*(N_ARGS_PLANE + N_ARGS_TRANS), 1, 1, 1, 1, 0, 0, 1, *[0]*N_ARGS_FINISH_PARAM, *[0]*N_ARGS_SELECT_PARAM],  # Ext
    [*[0]*N_ARGS_SKETCH, *[1]*(N_ARGS_PLANE + N_ARGS_TRANS), 0, 0, 0, 0, 1, 1, 1, *[0]*N_ARGS_FINISH_PARAM, *[0]*N_ARGS_SELECT_PARAM],  # Rev
    [*[0]*N_ARGS_SKETCH, *[1]*(N_ARGS_PLANE + N_ARGS_TRANS), 1, 1, 1, 1, 0, 0, 0, *[0]*N_ARGS_FINISH_PARAM, *[0]*N_ARGS_SELECT_PARAM],  # Pocket
    [*[0]*N_ARGS_SKETCH, *[1]*(N_ARGS_PLANE + N_ARGS_TRANS), 0, 0, 0, 0, 1, 1, 0, *[0]*N_ARGS_FINISH_PARAM, *[0]*N_ARGS_SELECT_PARAM],  # Groove
    [*[0]*N_ARGS_SKETCH, *[0]*N_ARGS_EXT, 1, 1, 0, 0, 0, 0, 0, 0, 0, *[0]*N_ARGS_SELECT_PARAM],  # Shell
    [*[0]*N_ARGS_SKETCH, *[0]*N_ARGS_EXT, 0, 0, 1, 1, 0, 0, 0, 0, 0, *[0]*N_ARGS_SELECT_PARAM],  # Chamfer
    [*[0]*N_ARGS_SKETCH, *[0]*N_ARGS_EXT, 0, 0, 0, 0, 1, 0, 0, 0, 0, *[0]*N_ARGS_SELECT_PARAM],  # Fillet
    [*[0]*N_ARGS_SKETCH, *[0]*N_ARGS_EXT, 0, 0, 0, 0, 0, 1, 0, 0, 0, *[0]*N_ARGS_SELECT_PARAM],  # Draft
    [*[0]*N_ARGS_SKETCH, *[0]*N_ARGS_EXT, *[0]*N_ARGS_FINISH_PARAM, *[0]*N_ARGS_SELECT_PARAM],   # Mirror
    [1, 1, 0, 0, 0, *[0]*N_ARGS_EXT, 0, 0, 0, 0, 0, 0, 1, 1, 1, *[0]*N_ARGS_SELECT_PARAM],  # Hole
    [*[0]*N_ARGS_SKETCH, *[0]*N_ARGS_EXT, *[0]*N_ARGS_FINISH_PARAM, *[0]*N_ARGS_SELECT_PARAM],   # TOPO
    [*[0]*N_ARGS_SKETCH, *[0]*N_ARGS_EXT, *[0]*N_ARGS_FINISH_PARAM, *[1]*N_ARGS_SELECT_PARAM],   # Select
    [*[0]*N_ARGS_SKETCH, *[0]*N_ARGS_EXT, *[0]*N_ARGS_FINISH_PARAM, *[0]*N_ARGS_SELECT_PARAM],   # MirrorStart
    [*[0]*N_ARGS_SKETCH, *[0]*N_ARGS_EXT, *[0]*N_ARGS_FINISH_PARAM, *[0]*N_ARGS_SELECT_PARAM],   # NoSharedIncluded
    [*[0]*N_ARGS_SKETCH, *[0]*N_ARGS_EXT, *[0]*N_ARGS_FINISH_PARAM, *[0]*N_ARGS_SELECT_PARAM],   # NoSharedIncludedEnd
    [*[0]*N_ARGS_SKETCH, *[0]*N_ARGS_EXT, *[0]*N_ARGS_FINISH_PARAM, *[0]*N_ARGS_SELECT_PARAM],   # AllOrientedIncluded1
    [*[0]*N_ARGS_SKETCH, *[0]*N_ARGS_EXT, *[0]*N_ARGS_FINISH_PARAM, *[0]*N_ARGS_SELECT_PARAM],   # AllOrientedIncluded2
    [*[0]*N_ARGS_SKETCH, *[0]*N_ARGS_EXT, *[0]*N_ARGS_FINISH_PARAM, *[0]*N_ARGS_SELECT_PARAM],   # AllOrientedIncludedEnd
    [*[0]*N_ARGS_SKETCH, *[0]*N_ARGS_EXT, *[0]*N_ARGS_FINISH_PARAM, *[0]*N_ARGS_SELECT_PARAM],   # AllPartiallySharedIncluded
    [*[0]*N_ARGS_SKETCH, *[0]*N_ARGS_EXT, *[0]*N_ARGS_FINISH_PARAM, *[0]*N_ARGS_SELECT_PARAM],   # AllPartiallySharedIncludedEnd
])

ARGS_N = 256  # 你的整数量化范围
