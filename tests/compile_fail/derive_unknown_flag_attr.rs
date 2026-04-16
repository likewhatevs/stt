#[derive(ktstr::Scheduler)]
#[scheduler(name = "bad")]
#[allow(dead_code)]
enum UnknownFlagAttrFlag {
    #[flag(bogus = ["x"])]
    Bad,
}

fn main() {}
