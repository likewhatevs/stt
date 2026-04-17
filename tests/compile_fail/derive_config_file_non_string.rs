#[derive(ktstr::Scheduler)]
#[scheduler(name = "bad", config_file = 42)]
#[allow(dead_code)]
enum ConfigFileNonStringFlag {}

fn main() {}
