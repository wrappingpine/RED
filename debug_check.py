import sys
sys.path.insert(0, '.')
from airmouse.tests.test_directional_coordination import create_mock_face, create_pipeline, MockHand
from airmouse.vision.head_coords import HeadCoordinateSystem
import numpy as np

projector, cursor, face = create_pipeline()

# Check the head coordinate system axes
hc = HeadCoordinateSystem.from_face(face)
print('Head coords:')
print('  Forward:', hc.forward)
print('  Right:', hc.right)
print('  Up:', hc.up)

# Check virtual plane
print('Virtual plane:')
print('  plane_right_head (X axis):', projector.virtual_plane._plane_x_axis_cam)
print('  plane_up_head (Y axis):', projector.virtual_plane._plane_y_axis_cam)

# Check camera_to_head transform
T = hc.get_transform_matrix()
print('Camera->Head transform matrix:')
print(T)

# Test projection with real hand positions
# Face: eye at (0.5, 0.5, 0), nose at (0.5, 0.5, -0.1) - forward is -Z
# Hand should be below face (Y > 0.5 in camera coords = down)
# and in front of face (Z < 0)

# Center hand at (0.5, 0.6, -0.1) - below eye, in front of nose
center_hand = MockHand(index_tip_x=0.5, index_tip_y=0.6, index_tip_z=-0.1)
result = projector.project(center_hand, face)
print('Center:', result.valid, result.u, result.v)

# Initialize cursor
cursor.get_relative_movement_from_plane(result.u, result.v)

# Move hand RIGHT (increase X)
right_hand = MockHand(index_tip_x=0.6, index_tip_y=0.6, index_tip_z=-0.1)
result = projector.project(right_hand, face)
print('Right:', result.valid, result.u, result.v)
movement = cursor.get_relative_movement_from_plane(result.u, result.v)
print('Movement RIGHT:', movement, 'dx > 0:', movement[0] > 0 if movement else None)

# Move hand LEFT (decrease X)
left_hand = MockHand(index_tip_x=0.4, index_tip_y=0.6, index_tip_z=-0.1)
result = projector.project(left_hand, face)
print('Left:', result.valid, result.u, result.v)
movement = cursor.get_relative_movement_from_plane(result.u, result.v)
print('Movement LEFT:', movement, 'dx < 0:', movement[0] < 0 if movement else None)

# Move hand UP (decrease Y - up in camera coords)
up_hand = MockHand(index_tip_x=0.5, index_tip_y=0.5, index_tip_z=-0.1)
result = projector.project(up_hand, face)
print('Up:', result.valid, result.u, result.v)
movement = cursor.get_relative_movement_from_plane(result.u, result.v)
print('Movement UP:', movement, 'dy < 0:', movement[1] < 0 if movement else None)

# Move hand DOWN (increase Y)
down_hand = MockHand(index_tip_x=0.5, index_tip_y=0.7, index_tip_z=-0.1)
result = projector.project(down_hand, face)
print('Down:', result.valid, result.u, result.v)
movement = cursor.get_relative_movement_from_plane(result.u, result.v)
print('Movement DOWN:', movement, 'dy > 0:', movement[1] > 0 if movement else None)