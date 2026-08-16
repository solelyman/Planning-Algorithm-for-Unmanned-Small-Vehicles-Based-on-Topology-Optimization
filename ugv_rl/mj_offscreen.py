"""
MuJoCo glfw 离屏渲染器 (动态障碍可见) — 照抄 mujoco-learning-main/get_camera_pic.py 成功配置
  关键差异 (vs mujoco.Renderer 高层 API 全黑):
  1. glfw 离屏窗口 + make_context_current (必须每帧渲染前切到本窗口 context)
  2. mjr_setBuffer(mjFB_OFFSCREEN) + mjr_render + mjr_readPixels 底层 API
  3. 相机用 mjCAMERA_TRACKING 跟车 (get_camera_pic.py 已验证画面变化)
"""
import glfw
import mujoco
import numpy as np

_glfw_initialized = False
_window = None


def _ensure_glfw():
    global _glfw_initialized, _window
    if not _glfw_initialized:
        glfw.init()
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        _window = glfw.create_window(640, 480, "offscreen", None, None)
        _glfw_initialized = True
        glfw.make_context_current(_window)


class MuJoCoOffscreenRenderer:
    def __init__(self, model, cam_name="fisheye", height=48, width=64):
        _ensure_glfw()
        self.model = model
        self.H = height
        self.W = width
        self.cam = mujoco.MjvCamera()
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        if cam_id >= 0:
            self.cam.fixedcamid = cam_id
            # TRACKING 跟车: 参考项目 get_camera_pic.py 用的模式, 画面会随车动
            self.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "chassis")
            self.cam.trackbodyid = body_id if body_id >= 0 else 1
            self.cam.distance = 0.0
            self.cam.azimuth = 0
            self.cam.elevation = 0
        self.scn = mujoco.MjvScene(model, maxgeom=2000)
        self.ctx = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_100.value)
        self.vopt = mujoco.MjvOption()
        self.pert = mujoco.MjvPerturb()
        self.viewport = mujoco.MjrRect(0, 0, self.W, self.H)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.ctx)
        self._buf = np.zeros((self.H, self.W, 3), dtype=np.uint8)

    def render(self, data):
        """渲染当前 data 的相机视图, 返回 HxWx3 RGB (0-255)"""
        glfw.make_context_current(_window)          # 关键: 每帧切回离屏 context
        mujoco.mj_forward(self.model, data)
        mujoco.mjv_updateScene(self.model, data, self.vopt, self.pert,
                               self.cam, mujoco.mjtCatBit.mjCAT_ALL.value, self.scn)
        mujoco.mjr_render(self.viewport, self.scn, self.ctx)
        mujoco.mjr_readPixels(self._buf, None, self.viewport, self.ctx)
        return np.flipud(self._buf)  # mjr_readPixels 是 bottom-up

    def render_gray(self, data):
        """返回 HxW float32 灰度 (0~255)"""
        rgb = self.render(data)
        return (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.uint8)
