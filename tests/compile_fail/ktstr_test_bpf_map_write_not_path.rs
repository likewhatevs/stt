use anyhow::Result;
use ktstr::assert::AssertResult;
use ktstr::ktstr_test;
use ktstr::scenario::Ctx;

#[ktstr_test(bpf_map_write = 42)]
fn bad(ctx: &Ctx) -> Result<AssertResult> {
    let _ = ctx;
    Ok(AssertResult::pass())
}

fn main() {}
