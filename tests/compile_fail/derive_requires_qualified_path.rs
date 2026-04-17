#[derive(ktstr::Scheduler)]
#[scheduler(name = "bad")]
#[allow(dead_code)]
enum RequiresQualifiedFlag {
    Alpha,
    #[flag(requires = [self::Alpha])]
    Beta,
}

fn main() {}
