# modelling motion ----simplified for now
# will later use omni.isaac.moveit or so...

from pxr import UsdGeom, Gf     # geometry tools in USD, and math library for vecotrs/ transforms
from omni.isaac.core.utils.stage import get_current_stage  # to get current stage(the whole simulation world)
import omni.timeline        # to control the simulation time (play, pause, reset)


class MotionInterface:    
    # moves the end effector up and down
    def __init__(self, flange_path):   # flange is where the end effector is
        self.stage = get_current_stage()   # get current stage
        self.flange_prim = self.stage.GetPrimAtPath(flange_path)  # get the end effector's prim

        if not self.flange_prim.IsValid():   # check if the flange_prim is valid
            raise RuntimeError(f"Invalid flange path: {flange_path}")

        self.xform = UsdGeom.Xformable(self.flange_prim)  # makes the object transformable(movable, rotatable)

    def move_up(self, distance=0.05):
        """
        Move end-effector up in Z direction
        """
        translate_ops = self.xform.GetOrderedXformOps()   # get the transform operations applied to the object

        if not translate_ops:   # check if the transform ops are valid
            print("[Motion] No transform ops found")
            return

        op = translate_ops[0]   # take the translation opertor
        current = op.Get()  # read current position

        new_pos = Gf.Vec3d(current[0], current[1], current[2] + distance)  # create new position ( change only Z)
        op.Set(new_pos)  # set the new position

        print(f"[Motion] Moved to {new_pos}")