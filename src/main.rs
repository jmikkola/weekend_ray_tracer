extern crate png;
extern crate rand;

use std::f64;
use std::fs::File;
use std::io::BufWriter;
use std::ops;
use std::path::Path;
use std::time::SystemTime;

use png::HasParameters;
use rand::prelude::*;

#[derive(Copy, Clone)]
struct Vec3 {
    x: f64,
    y: f64,
    z: f64,
}

impl Vec3 {
    fn new(x: f64, y: f64, z: f64) -> Self {
        Vec3 { x: x, y: y, z: z }
    }

    fn new_uniform(val: f64) -> Self {
        Self::new(val, val, val)
    }

    fn squared_length(self) -> f64 {
        self.x * self.x +
            self.y * self.y +
            self.z * self.z
    }

    fn length(self) -> f64 {
        self.squared_length().sqrt()
    }

    fn make_unit_vector(self) -> Self {
        self / Self::new_uniform(self.length())
    }

    fn unit_vector(val: f64) -> Self {
        Self::new_uniform(val).make_unit_vector()
    }

    fn add(self, val: f64) -> Self {
        self + Self::new_uniform(val)
    }

    fn mul(self, val: f64) -> Self {
        self * Self::new_uniform(val)
    }

    fn div(self, val: f64) -> Self {
        self / Self::new_uniform(val)
    }

    fn dot(self, v2: Self) -> f64 {
        self.x * v2.x + self.y * v2.y + self.z * v2.z
    }

    fn cross(self, v2: Self) -> Self {
        Self::new(
            self.y * v2.z - self.z * v2.y,
            -(self.x * v2.z - self.z * v2.x),
            self.x * v2.y - self.y * v2.x,
        )
    }
}

impl ops::Add for Vec3 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Vec3 { x: self.x + rhs.x, y: self.y + rhs.y, z: self.z + rhs.z, }
    }
}

impl ops::AddAssign for Vec3 {
    fn add_assign(&mut self, other: Vec3) {
        *self = *self + other
    }
}

impl ops::Sub for Vec3 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Vec3 { x: self.x - rhs.x, y: self.y - rhs.y, z: self.z - rhs.z, }
    }
}

impl ops::Mul for Vec3 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        Vec3 { x: self.x * rhs.x, y: self.y * rhs.y, z: self.z * rhs.z, }
    }
}

impl ops::Div for Vec3 {
    type Output = Self;

    fn div(self, rhs: Self) -> Self {
        Vec3 { x: self.x / rhs.x, y: self.y / rhs.y, z: self.z / rhs.z, }
    }
}

fn random_in_unit_sphere(rng: &mut ThreadRng) -> Vec3 {
    loop {
        let x: f64 = rng.gen();
        let y: f64 = rng.gen();
        let z: f64 = rng.gen();
        let p = Vec3::new(x, y, z).mul(2.0) - Vec3::new_uniform(1.0);
        if p.squared_length() < 1.0 {
            return p;
        }
    }
}

#[derive(Clone, Copy)]
struct Ray {
    a: Vec3,
    b: Vec3,
}

impl Ray {
    fn new(a: Vec3, b: Vec3) -> Self {
        Ray { a: a, b: b }
    }

    fn origin(self) -> Vec3 {
        self.a
    }

    fn direction(self) -> Vec3 {
        self.b
    }

    fn point_at_parameter(self, t: f64) -> Vec3 {
        self.a + self.b.mul(t)
    }
}

#[derive(Clone, Copy)]
struct HitRecord {
    t: f64,
    p: Vec3,
    normal: Vec3,
}

trait Hitable {
    fn hit(&self, r: Ray, t_min: f64, t_max: f64) -> Option<HitRecord>;
}

#[derive(Clone)]
struct Sphere {
    center: Vec3,
    radius: f64,
}

impl Sphere {
    fn new(center: Vec3, radius: f64) -> Self {
        Sphere { center: center, radius: radius }
    }
}

impl Hitable for Sphere {
    fn hit(&self, r: Ray, t_min: f64, t_max: f64) -> Option<HitRecord> {
        let oc = r.origin() - self.center;
        let a = r.direction().dot(r.direction());
        let b = oc.dot(r.direction());
        let c = oc.dot(oc) - self.radius*self.radius;
        let discriminant = b*b - a*c;
        if discriminant > 0.0 {
            let t1 = (-b - (b*b - a*c).sqrt()) / a;
            if t1 < t_max && t1 > t_min {
                let p = r.point_at_parameter(t1);
                return Some(HitRecord{
                    t: t1,
                    p: p,
                    normal: (p - self.center).div(self.radius),
                });
            }

            let t2 = (-b + (b*b - a*c).sqrt()) / a;
            if t2 < t_max && t2 > t_min {
                let p = r.point_at_parameter(t2);
                return Some(HitRecord{
                    t: t2,
                    p: p,
                    normal: (p - self.center).div(self.radius),
                });
            }
        }

        None
    }
}

struct HitableList {
    hitables: Vec<Box<Hitable>>,
}

impl Hitable for HitableList {
    fn hit(&self, r: Ray, t_min: f64, t_max: f64) -> Option<HitRecord> {
        let mut closest_so_far = t_max;
        let mut closest_hit = None;
        for ref item in &self.hitables {
            if let Some(hit) = item.hit(r, t_min, closest_so_far) {
                closest_so_far = hit.t;
                closest_hit = Some(hit);
            }
        }
        closest_hit
    }
}

struct Camera {
    origin: Vec3,
    lower_left_corner: Vec3,
    horizontal: Vec3,
    vertical: Vec3,
}

impl Camera {
    fn new() -> Self {
        Camera {
            origin: Vec3::new(0.0, 0.0, 0.0),
            lower_left_corner: Vec3::new(-2.0, -1.0, -1.0),
            horizontal: Vec3::new(4.0, 0.0, 0.0),
            vertical: Vec3::new(0.0, 2.0, 0.0),
        }
    }

    fn get_ray(&self, u: f64, v: f64) -> Ray {
        let direction = self.lower_left_corner + self.horizontal.mul(u) + self.vertical.mul(v);
        Ray::new(self.origin, direction)
    }
}

fn color<T>(r: Ray, world: &T, rng: &mut ThreadRng) -> Vec3 where T: Hitable {
    if let Some(hit) = &world.hit(r, 0.001, f64::MAX) {
        let target = hit.p + hit.normal + random_in_unit_sphere(rng);
        let r = Ray::new(hit.p, target - hit.p);
        color(r, world, rng).mul(0.5)
    } else {
        let unit_direction = r.direction().make_unit_vector();
        let t = 0.5 * (unit_direction.y + 1.0);
        Vec3::unit_vector(1.0).mul(1.0 - t) + Vec3::new(0.5, 0.7, 1.0).mul(t)
    }
}

fn render() -> (u32, u32, Vec<u8>) {
    let nx = 400;
    let ny = 200;

    let ns = 100; // samples

    let mut image = vec![];

    let lower_left_corner = Vec3::new(-2.0, -1.0, -1.0);
    let horizontal = Vec3::new(4.0, 0.0, 0.0);
    let vertical = Vec3::new(0.0, 2.0, 0.0);
    let origin = Vec3::new(0.0, 0.0, 0.0);

    let world = HitableList{
        hitables: vec![
            Box::new(Sphere::new(Vec3::new(0.0, 0.0, -1.0), 0.5)),
            Box::new(Sphere::new(Vec3::new(0.0, -100.5, -1.0), 100.)),
        ],
    };

    let cam = Camera::new();

    let mut rng = rand::thread_rng();

    for j in (0..ny).rev() {
        for i in 0..nx {

            let mut col = Vec3::new(0.0, 0.0, 0.0);

            for _ in 0..ns {
                let ir: f64 = rng.gen();
                let jr: f64 = rng.gen();
                let u = (i as f64 + ir) / nx as f64;
                let v = (j as f64 + jr) / ny as f64;
                let r = cam.get_ray(u, v);
                let p = r.point_at_parameter(2.0);

                col += color(r, &world, &mut rng);
            }

            let col2 = col.div(ns as f64);

            image.push((col2.x.sqrt() * 255.99) as u8);
            image.push((col2.y.sqrt() * 255.99) as u8);
            image.push((col2.z.sqrt() * 255.99) as u8);
            image.push(255); // alpha
        }
    }

    (nx, ny, image)
}

fn write_png(x: u32, y: u32, image: Vec<u8>, path: &Path) {
    let file = File::create(path).unwrap();
    let ref mut w = BufWriter::new(file);

    let mut encoder = png::Encoder::new(w, x, y);
    encoder.set(png::ColorType::RGBA).set(png::BitDepth::Eight);
    let mut writer = encoder.write_header().unwrap();

    writer.write_image_data(&image[..]).unwrap();
}

fn main() {
    let now_s = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let filename = format!("render_{}.png", now_s);
    let path = Path::new(&filename);
    let (x, y, image) = render();
    write_png(x, y, image, &path);
}
