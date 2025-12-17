use crate::engine::Engine;

mod types;
mod engine;

fn main() {
    let mut engine = Engine::new();
    let rid = engine.declare();
    let riid = engine.instantiate(rid);
    let eid = engine.spawn();
    let oid = engine.give(eid, riid);
    engine.take(eid, riid)
}
