use kiss3d::camera::Camera;
use kiss3d::event::WindowEvent;
use kiss3d::nalgebra::{Isometry3, Matrix4, Point3, Vector3};
use kiss3d::resource::ShaderUniform;
use kiss3d::window::Canvas;

/// A camera that cannot move.
#[derive(Clone, Debug)]
pub struct UserPerspectiveView {
    pa: Vector3<f32>,
    pb: Vector3<f32>,
    pc: Vector3<f32>,
    pe: Vector3<f32>,
    znear: f32,
    zfar: f32,
    proj: Matrix4<f32>,
    inv_proj: Matrix4<f32>,
}

impl UserPerspectiveView {
    pub fn new() -> Self {
        let w = 35.0;
        let h = 22.0;
        Self {
            pa: Vector3::new(-w / 2.0, -h / 2.0, 0.0),
            pb: Vector3::new(w / 2.0, -h / 2.0, 0.0),
            pc: Vector3::new(-w / 2.0, h / 2.0, 0.0),
            pe: Vector3::new(0.0, 0.0, 0.0),
            znear: 0.01,
            zfar: 1000.0,
            proj: nalgebra::one(),
            inv_proj: nalgebra::one(),
        }
    }

    pub(crate) fn set_eye_position(&mut self, pe: Vector3<f32>) {
        self.pe = pe;
        self.update_projviews();
    }

    /// Computes the projection matrix for a non-perpendicular viewing frustum in 3D space.
    ///
    /// This function calculates the projection matrix based on the given input parameters:
    /// `pa`, `pb`, and `pc` are the 3D coordinates of three points defining corners of the screen,
    /// `pe` is the 3D coordinate of the eye (camera) position, `n` is the distance from the eye to the
    /// near clipping plane, and `f` is the distance from the eye to the far clipping plane.
    ///
    /// The resulting projection matrix can be used to transform 3D coordinates from world space to
    /// clip space.
    ///
    /// # Arguments
    ///
    /// * `pa` - A reference to a `Vector3<f32>` representing the lower left screen corner
    /// * `pb` - A reference to a `Vector3<f32>` representing the lower right screen corner
    /// * `pc` - A reference to a `Vector3<f32>` representing the upper left screen corner
    /// * `pe` - A reference to a `Vector3<f32>` representing the eye (camera) position
    /// * `n` - The distance from the eye to the near clipping plane as a `f32`
    /// * `f` - The distance from the eye to the far clipping plane as a `f32`
    ///
    /// # Returns
    ///
    /// * A `Matrix4<f32>` representing the computed projection matrix
    ///
    /// # Example
    ///
    /// ```
    /// use nalgebra::{Matrix4, Vector3};
    ///
    /// let pa = Vector3::new(0.0, 0.0, 0.0);
    /// let pb = Vector3::new(1.0, 0.0, 0.0);
    /// let pc = Vector3::new(0.0, 1.0, 0.0);
    /// let pe = Vector3::new(0.0, 0.0, 5.0);
    /// let n = 1.0;
    /// let f = 1000.0;
    ///
    /// let projection_matrix = projection(&pa, &pb, &pc, &pe, n, f);
    /// ```
    fn projection(
        pa: &Vector3<f32>,
        pb: &Vector3<f32>,
        pc: &Vector3<f32>,
        pe: &Vector3<f32>,
        n: f32,
        f: f32,
    ) -> Matrix4<f32> {
        // Compute an orthonormal basis for the screen.
        let mut vr = pb - pa;
        let mut vu = pc - pa;

        vr.normalize_mut();
        vu.normalize_mut();

        let vn = vr.cross(&vu);
        let vn_norm = vn.normalize();

        // Compute the screen corner vectors.
        let va = pa - pe;
        let vb = pb - pe;
        let vc = pc - pe;

        // Find the distance from the eye to screen plane.
        let d = -va.dot(&vn_norm);

        // Find the extent of the perpendicular projection.
        let l = vr.dot(&va) * n / d;
        let r = vr.dot(&vb) * n / d;
        let b = vu.dot(&va) * n / d;
        let t = vu.dot(&vc) * n / d;

        // Create the perpendicular projection matrix.
        let mut proj_matrix = Matrix4::new(
            (2.0 * n) / (r - l),
            0.0,
            (r + l) / (r - l),
            0.0,
            0.0,
            (2.0 * n) / (t - b),
            (t + b) / (t - b),
            0.0,
            0.0,
            0.0,
            -(f + n) / (f - n),
            -(2.0 * f * n) / (f - n),
            0.0,
            0.0,
            -1.0,
            0.0,
        );

        // Rotate the projection to be non-perpendicular.
        let rot_matrix = Matrix4::new(
            vr.x, vr.y, vr.z, 0.0, vu.x, vu.y, vu.z, 0.0, vn.x, vn.y, vn.z, 0.0, 0.0, 0.0, 0.0, 1.0,
        );
        proj_matrix *= rot_matrix;

        // Move the apex of the frustum to the origin.
        proj_matrix *= Matrix4::new_translation(&-pe);
        proj_matrix
    }

    fn update_projviews(&mut self) {
        self.proj = Self::projection(
            &self.pa, &self.pb, &self.pc, &self.pe, self.znear, self.zfar,
        );
        let _ = self
            .proj
            .try_inverse()
            .map(|inv_proj| self.inv_proj = inv_proj);
    }
}

impl Camera for UserPerspectiveView {
    fn handle_event(&mut self, _: &Canvas, event: &WindowEvent) {
        //TODO: handle other events in the match statement
        match *event {
            WindowEvent::FramebufferSize(_, _) => {
                // TODO: Update to account for window width and height
                self.update_projviews();
            }
            _ => {}
        }
    }

    fn eye(&self) -> Point3<f32> {
        Point3::new(self.pe.x, self.pe.y, self.pe.z)
    }

    fn view_transform(&self) -> Isometry3<f32> {
        //TODO: this should be something different probably but we aren't using it
        Isometry3::identity()
    }

    fn transformation(&self) -> Matrix4<f32> {
        self.proj
    }

    fn inverse_transformation(&self) -> Matrix4<f32> {
        self.inv_proj
    }

    fn clip_planes(&self) -> (f32, f32) {
        (self.znear, self.zfar)
    }

    fn update(&mut self, _: &Canvas) {}

    #[inline]
    fn upload(
        &self,
        _: usize,
        proj: &mut ShaderUniform<Matrix4<f32>>,
        view: &mut ShaderUniform<Matrix4<f32>>,
    ) {
        let view_mat = Matrix4::identity();
        proj.upload(&self.proj);
        view.upload(&view_mat);
    }
}
