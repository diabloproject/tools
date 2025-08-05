use std::pin::Pin;

use tower::Service;
use crate::proto::*;


// struct L1Server {
//     // handler: S
// }


// impl<Q: PayloadConvertable, S: PayloadConvertable> Service<L1Packet<Q>> for L1Server {
//     type Response = L1Packet<S>;

//     type Error = PacketDecodeError;

//     type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

//     fn poll_ready(&mut self, cx: &mut std::task::Context<'_>) -> std::task::Poll<Result<(), Self::Error>> {

//     }

//     fn call(&mut self, req: L1Packet<Q>) -> Self::Future {
//         todo!()
//     }
// }
