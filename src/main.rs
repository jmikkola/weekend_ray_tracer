extern crate png;
extern crate rand;
extern crate rayon;

use core::f32::consts::PI;
use std::f32;
use std::fs::File;
use std::io::BufWriter;
use std::ops;
use std::path::Path;
use std::time::SystemTime;

use png::HasParameters;
use rand::prelude::*;
use rayon::prelude::*;

#[derive(Copy, Clone)]
struct Vec3 {
    x: f32,
    y: f32,
    z: f32,
}

impl Vec3 {
    fn new(x: f32, y: f32, z: f32) -> Self {
        Vec3 { x: x, y: y, z: z }
    }

    fn new_uniform(val: f32) -> Self {
        Self::new(val, val, val)
    }

    fn squared_length(self) -> f32 {
        self.x * self.x +
            self.y * self.y +
            self.z * self.z
    }

    fn length(self) -> f32 {
        self.squared_length().sqrt()
    }

    fn make_unit_vector(self) -> Self {
        self / Self::new_uniform(self.length())
    }

    fn unit_vector(val: f32) -> Self {
        Self::new_uniform(val).make_unit_vector()
    }

    fn add(self, val: f32) -> Self {
        self + Self::new_uniform(val)
    }

    fn mul(self, val: f32) -> Self {
        self * Self::new_uniform(val)
    }

    fn div(self, val: f32) -> Self {
        self / Self::new_uniform(val)
    }

    fn dot(self, v2: Self) -> f32 {
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
        let x: f32 = rng.gen();
        let y: f32 = rng.gen();
        let z: f32 = rng.gen();
        let p = Vec3::new(x, y, z).mul(2.0) - Vec3::new_uniform(1.0);
        if p.squared_length() < 1.0 {
            return p;
        }
    }
}

fn random_in_unit_disk(rng: &mut ThreadRng) -> Vec3 {
    loop {
        let r1: f32 = rng.gen();
        let r2: f32 = rng.gen();
        let p = Vec3::new(r1, r2, 0.0).mul(2.0) - Vec3::new(1.0, 1.0, 0.0);
        if p.dot(p) >= 1.0 {
            return p;
        }
    }
}

fn random_in_unit_disk2(rng: &mut ThreadRng) -> Vec3 {
    let v = random_in_unit_disk(rng);
    v * v
}

fn reflect(v: Vec3, n: Vec3) -> Vec3 {
    v - n.mul(2.0 * v.dot(n))
}

fn refract(v: Vec3, n: Vec3, ni_over_nt: f32) -> Option<Vec3> {
    let uv = v.make_unit_vector();
    let dt = uv.dot(n);
    let discriminant = 1.0 - ni_over_nt * ni_over_nt * (1.0 - dt*dt);
    if discriminant > 0.0 {
        Some((uv - n.mul(dt)).mul(ni_over_nt) - n.mul(discriminant.sqrt()))
    } else {
        None
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

    fn point_at_parameter(self, t: f32) -> Vec3 {
        self.a + self.b.mul(t)
    }
}

struct HitRecord {
    t: f32,
    p: Vec3,
    normal: Vec3,
    material: Material,
}

trait Hitable {
    fn hit(&self, r: Ray, t_min: f32, t_max: f32) -> Option<HitRecord>;
}

struct Sphere {
    center: Vec3,
    radius: f32,
    material: Material,
}

impl Sphere {
    fn new(center: Vec3, radius: f32, material: Material) -> Self {
        Sphere { center: center, radius: radius, material: material, }
    }
}

impl Hitable for Sphere {
    fn hit(&self, r: Ray, t_min: f32, t_max: f32) -> Option<HitRecord> {
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
                    material: self.material,
                });
            }

            let t2 = (-b + (b*b - a*c).sqrt()) / a;
            if t2 < t_max && t2 > t_min {
                let p = r.point_at_parameter(t2);
                return Some(HitRecord{
                    t: t2,
                    p: p,
                    normal: (p - self.center).div(self.radius),
                    material: self.material,
                });
            }
        }

        None
    }
}

struct HitableList {
    hitables: Vec<Box<Hitable>>,
}

unsafe impl Sync for HitableList {
}

impl Hitable for HitableList {
    fn hit(&self, r: Ray, t_min: f32, t_max: f32) -> Option<HitRecord> {
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

#[derive(Clone, Copy)]
enum Material {
    Lambertian {
        albedo: Vec3,
    },
    Metal {
        albedo: Vec3,
        fuzz: f32,
    },
    Dielectrict {
        ref_idx: f32,
    },
}

fn schlick(cosine: f32, ref_idx: f32) -> f32 {
    let r0 = (1.0 - ref_idx) / (1.0 + ref_idx);
    let r1 = r0 * r0;
    r1 + (1.0 - r1) * (1.0 - cosine).powf(5.0)
}

impl Material {
    fn scatter(&self, r_in: Ray, hit: &HitRecord, rng: &mut ThreadRng) -> (Vec3, Ray, bool) {
        match self {
            Material::Lambertian{albedo} => {
                let target = hit.p + hit.normal + random_in_unit_sphere(rng);
                let scattered = Ray::new(hit.p, target - hit.p);
                (*albedo, scattered, true)
            },

            Material::Metal{albedo, fuzz} => {
                let reflected = reflect(r_in.direction().make_unit_vector(), hit.normal);
                let scattered = Ray::new(hit.p, reflected + random_in_unit_sphere(rng).mul(*fuzz));
                let b = scattered.direction().dot(hit.normal) > 0.0;
                (*albedo, scattered, b)
            },

            Material::Dielectrict{ref_idx} => {
                let reflected = reflect(r_in.direction(), hit.normal);
                let attenuation = Vec3::new_uniform(1.0);

                let mut outward_normal = Vec3::new_uniform(1.0);
                let mut ni_over_nt = 0.0;
                let mut cosine = 0.0;
                if r_in.direction().dot(hit.normal) > 0.0 {
                    outward_normal = hit.normal.mul(-1.0);
                    ni_over_nt = *ref_idx;
                    cosine = ref_idx * r_in.direction().dot(hit.normal) / r_in.direction().length();
                } else {
                    outward_normal = hit.normal;
                    ni_over_nt = 1.0 / *ref_idx;
                    cosine = -r_in.direction().dot(hit.normal) / r_in.direction().length();
                }

                if let Some(refracted) = refract(r_in.direction(), outward_normal, ni_over_nt) {
                    let reflect_prob = schlick(cosine, *ref_idx);
                    let rn: f32 = rng.gen();
                    let scattered = if rn < reflect_prob {
                        Ray::new(hit.p, reflected)
                    } else {
                        Ray::new(hit.p, refracted)
                    };
                    (attenuation, scattered, true)
                } else {
                    (attenuation, Ray::new(hit.p, reflected), true)
                }
            },
        }
    }
}

struct Camera {
    origin: Vec3,
    lower_left_corner: Vec3,
    horizontal: Vec3,
    vertical: Vec3,
    u: Vec3,
    v: Vec3,
    w: Vec3,

    lens_radius: f32,
}

impl Camera {
    fn new(lookfrom: Vec3, lookat: Vec3, vup: Vec3, vfov: f32, aspect: f32, aperture: f32, focus_dist: f32) -> Self {
        let theta = vfov * PI / 180.0;
        let half_height = (theta/2.0).tan();
        let half_width = aspect * half_height;

        let w = (lookfrom - lookat).make_unit_vector();
        let u = vup.cross(w).make_unit_vector();
        let v = w.cross(u);

        let origin = lookfrom;

        Camera {
            lens_radius: aperture / 2.0,

            u: u,
            v: v,
            w: w,

            origin: origin,
            lower_left_corner: origin - u.mul(half_width*focus_dist) - v.mul(half_height*focus_dist) - w.mul(focus_dist),
            horizontal: u.mul(2.0 * half_width * focus_dist),
            vertical: v.mul(2.0 * half_height * focus_dist),
        }
    }

    fn get_ray(&self, s: f32, t: f32, rng: &mut ThreadRng) -> Ray {
        let rd = random_in_unit_disk2(rng).mul(self.lens_radius);
        let offset = self.u.mul(rd.x) + self.v.mul(rd.y);
        let direction = self.lower_left_corner + self.horizontal.mul(s) + self.vertical.mul(t) - self.origin - offset;
        Ray::new(self.origin + offset, direction)
    }
}

fn color<T>(r: Ray, world: &T, depth: u32, rng: &mut ThreadRng) -> Vec3 where T: Hitable {
    if let Some(hit) = &world.hit(r, 0.001, f32::MAX) {
        let (attenuation, scattered, b) = hit.material.scatter(r, hit, rng);
        if depth < 10 && b {
            return attenuation * color(scattered, world, depth + 1, rng);
        } else {
            return Vec3::new(0.0, 0.0, 0.0);
        }
    } else {
        let unit_direction = r.direction().make_unit_vector();
        let t = 0.5 * (unit_direction.y + 1.0);
        Vec3::unit_vector(1.0).mul(1.0 - t) + Vec3::new(0.5, 0.7, 1.0).mul(t)
    }
}

fn gen64(rng: &mut ThreadRng) -> f32 {
    let r: f32 = rng.gen();
    r
}

fn random_scene() -> HitableList {
    let n = 500;
    let mut hitables: Vec<Box<Hitable>> = vec![
        Box::new(Sphere::new(
            Vec3::new(0.0, -1000.0, 0.0),
            1000.0,
            Material::Lambertian{
                albedo: Vec3::new(0.5, 0.5, 0.5),
            },
        )),
    ];

    let mut rng = rand::thread_rng();

    for a in -11..11 {
        for b in -11..11 {
            let choose_mat: f32 = rng.gen();
            let center = Vec3::new(
                a as f32 + 0.8 * gen64(&mut rng),
                0.2,
                b as f32 + 0.8 * gen64(&mut rng),
            );

            if (center - Vec3::new(4.0, 0.2, 0.0)).length() > 0.9 {
                if choose_mat < 0.60 {
                    // diffuse
                    hitables.push(Box::new(Sphere::new(
                        center,
                        0.2,
                        Material::Lambertian{
                            albedo: Vec3::new(
                                gen64(&mut rng) * gen64(&mut rng),
                                gen64(&mut rng) * gen64(&mut rng),
                                gen64(&mut rng) * gen64(&mut rng),
                            ),
                        },
                    )));
                } else if choose_mat < 0.90 {
                    // metal
                    hitables.push(Box::new(Sphere::new(
                        center,
                        0.2,
                        Material::Metal{
                            albedo: Vec3::new(
                                0.5 * (1.0 + gen64(&mut rng)),
                                0.5 * (1.0 + gen64(&mut rng)),
                                0.5 * (1.0 + gen64(&mut rng)),
                            ),
                            fuzz: 0.5 * gen64(&mut rng),
                        },
                    )));
                } else {
                    // glass
                    hitables.push(Box::new(Sphere::new(
                        center,
                        0.2,
                        Material::Dielectrict{
                            ref_idx: 1.5,
                        },
                    )));
                }
            }
        }
    }


    hitables.push(Box::new(Sphere::new(
        Vec3::new(0.0, 1.0, 0.0),
        1.0,
        Material::Dielectrict{ref_idx: 1.5,},
    )));


    hitables.push(Box::new(Sphere::new(
        Vec3::new(-4.0, 1.0, 0.0),
        1.0,
        Material::Lambertian{
            albedo: Vec3::new(0.4, 0.2, 0.1),
        },
    )));

    hitables.push(Box::new(Sphere::new(
        Vec3::new(4.0, 1.0, 0.0),
        1.0,
        Material::Metal{
            albedo: Vec3::new(0.7, 0.6, 0.5),
            fuzz: 0.0,
        },
    )));

    return HitableList{hitables: hitables};
}

fn render() -> (u32, u32, Vec<u8>) {
    let nx = 3440;
    let ny = 1440;

    let ns = 400; // samples

    let r = (PI/4.0).cos();

    let world1 = HitableList{
        hitables: vec![
            /*
            Box::new(Sphere::new(
                Vec3::new(-r, 0.0, -1.0),
                r,
                Material::Lambertian{
                   albedo: Vec3::new(0.0, 0.0, 1.0),
                },
            )),
            Box::new(Sphere::new(
                Vec3::new(r, 0.0, -1.0),
                r,
                Material::Lambertian{
                   albedo: Vec3::new(1.0, 0.0, 0.0),
                },
            )),
             */
            Box::new(Sphere::new(
                Vec3::new(0.0, 0.0, -1.0),
                0.5,
                Material::Lambertian{
                   albedo: Vec3::new(0.1, 0.2, 0.5),
                },
            )),
            Box::new(Sphere::new(
                Vec3::new(0.0, -100.5, -1.0),
                100.0,
                Material::Lambertian{
                   albedo: Vec3::new(0.8, 0.8, 0.0),
                },
            )),
            Box::new(Sphere::new(
                Vec3::new(1.0, 0.0, -1.0),
                0.5,
                Material::Metal{
                    albedo: Vec3::new(0.8, 0.6, 0.2),
                    fuzz: 0.3,
                },
            )),
            Box::new(Sphere::new(
                Vec3::new(-1.0, 0.0, -1.0),
                0.5,
                Material::Dielectrict{
                    ref_idx: 1.5,
                },
            )),
            Box::new(Sphere::new(
                Vec3::new(-1.0, 0.0, -1.0),
                -0.45,
                Material::Dielectrict{
                    ref_idx: 1.5,
                },
            )),
        ],
    };

    let world = random_scene();

    let lookfrom = Vec3::new(13.0, 2.0, 3.0);
    let lookat = Vec3::new(0.0, 0.0, 0.0);
    let dist_to_focus = 10.0;
    let aperture = 0.1;
    let cam = Camera::new(
        lookfrom,
        lookat,
        Vec3::new(0.0, 1.0, 0.0),
        20.0,
        nx as f32 / ny as f32,
        aperture,
        dist_to_focus,
    );

    let image = (0..ny).into_par_iter().map(|j| {
        let mut rng = rand::thread_rng();
        let mut row = vec![];

        println!("row {j}");

        for i in 0..nx {
            let mut col = Vec3::new(0.0, 0.0, 0.0);

            for _ in 0..ns {
                let ir: f32 = rng.gen();
                let jr: f32 = rng.gen();
                let u = (i as f32 + ir) / nx as f32;
                let v = (j as f32 + jr) / ny as f32;
                let r = cam.get_ray(u, v, &mut rng);
                let p = r.point_at_parameter(2.0);

                col += color(r, &world, 0, &mut rng);
            }

            let col2 = col.div(ns as f32);

            row.push((col2.x.sqrt() * 255.99) as u8);
            row.push((col2.y.sqrt() * 255.99) as u8);
            row.push((col2.z.sqrt() * 255.99) as u8);
            row.push(255); // alpha
        }

        return row;
    })
        .rev()
        .flatten()
        .collect();

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
