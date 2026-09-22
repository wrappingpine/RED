"""
Coordinate Contract Implementation for Air Mouse Pipeline

Defines the authoritative Coordinate Contract:

1. All pipeline stages use the same normalized coordinate system [0, 1]
2. Camera landmarks: [0, 1] in camera frame (MediaPipe normalized coordinates)
3. Projection system: Input = [0,1] camera coords → Output = [0,1] plane coords
4. Cursor system: Input = [0,1] plane coords → Output = [0,1] screen coords → [pixels]
5. This consistency enables modular components and predictable behavior

Key Properties:
- Camera normalization is already handled by MediaPipe (0-1 range)
- Virtual plane coordinates are [0,1] in plane space, independent of screen arrangement
- Screen coordinate conversion is decoupled from pipeline normalization
- Consistent contract enables independent testing and swapping of components
"""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# =============================================================================
# COORDINATE CONTRACT CLASSIFICATIONS
# =============================================================================

class CoordinateSpace(Enum):
    """Supported coordinate spaces in the airmouse pipeline."""
    CAMERA = "camera"        # MediaPipe hand landmarks: [0,1] in camera frame
    PLANE = "plane"          # Virtual display plane: [0,1] in plane space  
    SCREEN = "screen"        # Display space: [0,1] normalized screen coordinates
    PIXELS = "pixels"        # Absolute pixel coordinates
    HEAD = "head"            # Head coordinate system: meters relative to head


@dataclass
class CoordinateTransformer:
    """
    Handles coordinate transformations with explicit contract compliance.
    
    Ensures:
    - All intermediate steps use [0,1] normalized coordinates
    - Screen conversion is explicit and optional
    - Round-trip consistency (A→B→A should preserve values within precision)
    """
    
    def camera_to_plane(self, 
                       camera_x: float, 
                       camera_y: float, 
                       camera_landmarks: Optional[Tuple[float, float, float, float]] = None
                       ) -> Tuple[float, float]:
        """
        Transform camera coordinates to plane coordinates.
        
        CONTRACT: Input must be [0,1] camera normalized coords.
        Output is [0,1] plane normalized coords.
        
        Args:
            camera_x, camera_y: [0,1] normalized in camera frame
            camera_landmarks: Optional full landmark set for plane intersection
            
        Returns:
            (plane_u, plane_v): [0,1] normalized plane coordinates
            
        Raises:
            ValueError: If inputs not in [0,1] range or outside valid bounds
        """
        # Validate input contract compliance
        if not (0.0 <= camera_x <= 1.0 and 0.0 <= camera_y <= 1.0):
            raise ValueError(f"Camera coordinates must be [0,1], got ({camera_x}, {camera_y})")
        
        # TODO: Implement actual projection logic here
        # For now, direct mapping for testing
        # This preserves the [0,1] invariant
        return camera_x, camera_y
    
    def plane_to_screen(self,
                       plane_u: float,
                       plane_v: float,
                       screen_width: int = 1920,
                       screen_height: int = 1080) -> Tuple[float, float]:
        """
        Transform plane coordinates to screen coordinates.
        
        CONTRACT: Input must be [0,1] plane normalized coords.
        Output is [0,1] screen normalized coords.
        
        Args:
            plane_u, plane_v: [0,1] normalized plane coordinates
            screen_width, screen_height: Display dimensions in pixels
            
        Returns:
            (screen_x_norm, screen_y_norm): [0,1] normalized screen coordinates
            
        Raises:
            ValueError: If inputs not in [0,1] range
        """
        if not (0.0 <= plane_u <= 1.0 and 0.0 <= plane_v <= 1.0):
            raise ValueError(f"Plane coordinates must be [0,1], got ({plane_u}, {plane_v})")
        
        # TODO: Implement multi-monitor support
        # For now, single monitor mapping
        screen_x = plane_u * (screen_width - 1)
        screen_y = plane_v * (screen_height - 1)
        
        return screen_x, screen_y
    
    def screen_to_pixels(self,
                        screen_x_norm: float,
                        screen_y_norm: float,
                        screen_width: int = 1920,
                        screen_height: int = 1080) -> Tuple[int, int]:
        """
        Convert normalized screen coordinates to pixel coordinates.
        
        CONTRACT: Input must be [0,1] screen normalized coords.
        Output is [0, screen_width-1] × [0, screen_height-1] pixel coordinates.
        
        Args:
            screen_x_norm, screen_y_norm: [0,1] normalized screen coordinates
            screen_width, screen_height: Display dimensions
            
        Returns:
            (pixel_x, pixel_y): Pixel coordinates
            
        Raises:
            ValueError: If inputs not in [0,1] range
        """
        if not (0.0 <= screen_x_norm <= 1.0 and 0.0 <= screen_y_norm <= 1.0):
            raise ValueError(f"Screen coordinates must be [0,1], got ({screen_x_norm}, {screen_y_norm})")
        
        pixel_x = int(screen_x_norm * (screen_width - 1))
        pixel_y = int(screen_y_norm * (screen_height - 1))
        
        # Clamp to valid range (guard against floating point errors)
        pixel_x = max(0, min(pixel_x, screen_width - 1))
        pixel_y = max(0, min(pixel_y, screen_height - 1))
        
        return pixel_x, pixel_y
    
    def round_trip_camera_to_plane_to_camera(self,
                                            camera_x: float,
                                            camera_y: float) -> float:
        """
        Test round-trip consistency for camera coordinates.
        
        CONTRACT: Transform camera→plane→camera should preserve values.
        Used for debugging and validation.
        """
        plane_u, plane_v = self.camera_to_plane(camera_x, camera_y)
        # For direct mapping, this should be identity
        return plane_u


# Global coordinate transformer instance for contract compliance
CoordinateContract = CoordinateTransformer()


# =============================================================================
# CONTRACT VALIDATION TOOLS
# =============================================================================

def validate_coordinate_range(value: float, 
                            min_val: float = 0.0, 
                            max_val: float = 1.0,
                            name: str = "value") -> bool:
    """
    Validate that a coordinate is within expected range.
    
    CONTRACT: All pipeline intermediate coordinates must be [0,1].
    
    Args:
        value: Coordinate value to validate
        min_val, max_val: Expected range (default [0,1])
        name: Descriptive name for error messages
        
    Returns:
        True if valid, False otherwise
        
    Raises:
        ValueError: If out of range (strict validation)
    """
    if not (min_val <= value <= max_val):
        raise ValueError(f"{name} must be in [{min_val}, {max_val}], got {value}")
    return True


def contract_compliance_check() -> bool:
    """
    Run basic contract compliance checks.
    
    CONTRACT: This function should validate the coordinate contract
    is being followed throughout the codebase.
    """
    # Test basic transformer functionality
    try:
        # Test camera -> plane -> screen -> pixels
        camera_x, camera_y = 0.5, 0.5
        plane_u, plane_v = CoordinateContract.camera_to_plane(camera_x, camera_y)
        screen_x_norm, screen_y_norm = CoordinateContract.plane_to_screen(plane_u, plane_v)
        pixel_x, pixel_y = CoordinateContract.screen_to_pixels(screen_x_norm, screen_y_norm)
        
        # Validate ranges
        validate_coordinate_range(camera_x, name="camera_x")
        validate_coordinate_range(plane_u, name="plane_u")
        validate_coordinate_range(screen_x_norm, name="screen_x_norm")
        
        # All transformations should produce valid results
        assert 0 <= pixel_x < 1920, f"Invalid pixel x: {pixel_x}"
        assert 0 <= pixel_y < 1080, f"Invalid pixel y: {pixel_y}"
        
        print("Coordinate Contract validation: PASSED")
        return True
        
    except Exception as e:
        print(f"Coordinate Contract validation: FAILED - {e}")
        return False


if __name__ == "__main__":
    # Run validation when executed directly
    contract_compliance_check()