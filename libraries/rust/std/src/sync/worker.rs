use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;

pub enum WorkerMsg<T> {
    Item(T),
    Finish,
}

pub struct Worker<T> {
    tx: Sender<WorkerMsg<T>>,
    handle: Option<thread::JoinHandle<()>>,
}

impl<T: Send + 'static> Worker<T> {
    pub fn new(mut handler: impl FnMut(T) + Send + 'static) -> Self {
        let (tx, rx): (Sender<WorkerMsg<T>>, Receiver<WorkerMsg<T>>) = mpsc::channel();

        let handle = thread::spawn(move || {
            while let Ok(msg) = rx.recv() {
                match msg {
                    WorkerMsg::Item(value) => handler(value),
                    WorkerMsg::Finish => break,
                }
            }

            // Drain any late arrivals (extremely rare but safe)
            while let Ok(WorkerMsg::Item(value)) = rx.try_recv() {
                handler(value);
            }
        });

        Self {
            tx,
            handle: Some(handle),
        }
    }

    pub fn send(&self, value: T) -> Result<(), mpsc::SendError<WorkerMsg<T>>> {
        self.tx.send(WorkerMsg::Item(value))
    }
}

impl<T> Drop for Worker<T> {
    fn drop(&mut self) {
        // Send finish message — guaranteed to arrive after all sent items
        let _ = self.tx.send(WorkerMsg::Finish);

        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}
