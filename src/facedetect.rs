use opencv::{core, highgui, imgproc, objdetect, prelude::*, types, videoio, Result};

#[allow(dead_code)]
pub fn run() -> Result<()> {
    let mut camera = videoio::VideoCapture::new(0, videoio::CAP_ANY)?;
    let xml = "/Users/liam/Desktop/facetrack/src/haarcascade_frontalface_default.xml";
    let mut face_detector = objdetect::CascadeClassifier::new(xml)?;
    let mut img = Mat::default();

    loop {
        camera.read(&mut img)?;
        let face = get_face_location(&img, &mut face_detector)?;

        if let Some(face) = face {
            dbg!(face);
            imgproc::rectangle(
                &mut img,
                face,
                core::Scalar::new(0f64, 255f64, 0f64, 0f64),
                2,
                imgproc::LINE_8,
                0,
            )?;
        }
        highgui::imshow("gray", &img)?;
        highgui::wait_key(1)?;
    }
}

pub fn get_face_location(
    img: &Mat,
    face_detector: &mut objdetect::CascadeClassifier,
) -> Result<Option<core::Rect>> {
    let mut gray = Mat::default();
    imgproc::cvt_color(&img, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;

    let mut faces = types::VectorOfRect::new();
    face_detector.detect_multi_scale(
        &gray,
        &mut faces,
        1.1,
        3,
        objdetect::CASCADE_SCALE_IMAGE | objdetect::CASCADE_FIND_BIGGEST_OBJECT,
        core::Size::new(100, 100),
        core::Size::new(1000, 1000),
    )?;

    if !faces.is_empty() {
        Ok(Some(faces.get(0)?))
    } else {
        Ok(None)
    }
}
