extern crate kiss3d;

use crate::usercam::UserPerspectiveView;
use kiss3d::light::Light;
use kiss3d::nalgebra::{Point3, Translation3, Vector3};
use kiss3d::window::Window;
use opencv::{objdetect, prelude::*, videoio, Result};
use rand::Rng;

mod facedetect;
mod usercam;

fn main() -> Result<()> {
    // Can uncomment this to look at the facetracker itself
    // facedetect::run();

    // Initialize the camera and the face detector.
    let mut camera = videoio::VideoCapture::new(0, videoio::CAP_ANY)?;
    let xml = "./src/haarcascade_frontalface_default.xml";
    let mut face_detector = objdetect::CascadeClassifier::new(xml)?;
    let mut img = Mat::default();

    // Create the 3D window and a random number generator.
    let mut window = Window::new("Liam's Cube Demo");
    let mut rng = rand::thread_rng();

    // Cubes are 1cm in all dimensions
    let cube_size = 1.0;

    // Create 100 random cubes with random colors and positions.
    for _ in 0..100 {
        let x = rng.gen_range(-20.0..20.0);
        let y = rng.gen_range(-20.0..20.0);
        let z = rng.gen_range(-20.0..20.0);

        let mut c = window.add_cube(cube_size, cube_size, cube_size);
        c.set_color(
            rng.gen_range(0.0..1.0),
            rng.gen_range(0.0..1.0),
            rng.gen_range(0.0..1.0),
        );
        c.set_local_translation(Translation3::new(x, y, z));
    }

    // Set up the lighting for the 3D scene.
    window.set_light(Light::Absolute(Point3::new(50.0, 50.0, 90.0)));

    // Set up the user perspective camera.
    // Width and Height are in cm, screen centre is at 0,0,0
    let mut cam = UserPerspectiveView::new(35.0, 22.0);

    // Main render loop.
    let mut pe = Vector3::new(0.0, 0.0, 40.0);
    while window.render_with_camera(&mut cam) {
        // Read the current frame from the camera.
        camera.read(&mut img)?;

        // Detect the face in the current frame.
        if let Ok(Some(face)) = facedetect::get_face_location(&img, &mut face_detector) {
            // TODO: Refine the calculations for the eye position based on face location.
            // Units for translation are in centimeters.
            // Face units are in pixels.
            pe.x = (800.0 - face.x as f32) / 100.0;
            pe.y = (400.0 - face.y as f32) / 100.0;
            pe.z = 40.0 + (450.0 - face.width as f32) / 25.0;
        }

        // Update the camera's eye position.
        cam.set_eye_position(pe);
    }

    Ok(())
}
