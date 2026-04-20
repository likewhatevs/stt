use ktstr::ktstr_test;

// Array elements in `workloads = [..]` must be Payload paths, not
// literal values.
#[ktstr_test(workloads = [42])]
fn bad(_ctx: &ktstr::scenario::Ctx) -> anyhow::Result<ktstr::assert::AssertResult> {
    Ok(ktstr::assert::AssertResult::pass())
}

fn main() {}
