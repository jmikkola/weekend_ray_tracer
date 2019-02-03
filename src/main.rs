use std::ops;

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

    fn dot(self, v2: Self) -> Self {
        Self::new(self.x * v2.x, self.y * v2.y, self.z * v2.z)
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

fn color(r: Ray) -> Vec3 {
    let unit_direction = r.direction().make_unit_vector();
    let t = 0.5 * (unit_direction.y + 1.0);
    Vec3::unit_vector(1.0).mul(1.0 - t) + Vec3::new(0.5, 0.7, 1.0).mul(t)
}


fn main() {
    let nx = 300;
    let ny = 100;

    println!("P3");
    println!("{} {}", nx, ny);
    println!("255");

    let lower_left_corner = Vec3::new(-2.0, -1.0, -1.0);
    let horizontal = Vec3::new(4.0, 0.0, 0.0);
    let vertical = Vec3::new(0.0, 2.0, 0.0);
    let origin = Vec3::new(0.0, 0.0, 0.0);

    for j in (0..ny).rev() {
        for i in 0..nx {
            let u = i as f64 / nx as f64;
            let v = j as f64 / ny as f64;
            let r = Ray::new(origin, lower_left_corner + horizontal.mul(u) + vertical.mul(v));
            let col = color(r);

            let col2 = col.mul(255.99);
            println!("{} {} {}", col2.x as i64, col2.y as i64, col2.z as i64);
        }
    }
}
