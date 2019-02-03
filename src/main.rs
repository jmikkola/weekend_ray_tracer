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


fn main() {
    let nx = 300;
    let ny = 100;
    println!("P3");
    println!("{} {}", nx, ny);
    println!("255");
    for j in (0..ny).rev() {
        for i in 0..nx {
            let col = Vec3::new(i as f64 / nx as f64, j as f64 / nx as f64, 0.2);
            let col2 = col.mul(255.99);
            println!("{} {} {}", col2.x as i64, col2.y as i64, col2.z as i64);
        }
    }
}
