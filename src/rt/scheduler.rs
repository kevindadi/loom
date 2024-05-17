#![allow(deprecated)]

use crate::rt::Execution;

use generator::{self, Generator, Gn};
use scoped_tls::scoped_thread_local;
use std::cell::RefCell;
use std::collections::VecDeque;

pub(crate) struct Scheduler {
    max_threads: usize,
}

type Thread = Generator<'static, Option<Box<dyn FnOnce()>>, ()>;

scoped_thread_local! {
    static STATE: RefCell<State<'_>>
}

struct QueuedSpawn {
    f: Box<dyn FnOnce()>,
    stack_size: Option<usize>,
}

struct State<'a> {
    execution: &'a mut Execution,
    queued_spawn: &'a mut VecDeque<QueuedSpawn>,
}

impl Scheduler {
    /// Create an execution
    pub(crate) fn new(capacity: usize) -> Scheduler {
        Scheduler {
            max_threads: capacity,
        }
    }

    /// Access the execution
    pub(crate) fn with_execution<F, R>(f: F) -> R
    where
        F: FnOnce(&mut Execution) -> R,
    {
        Self::with_state(|state| f(state.execution))
    }

    /// Perform a context switch
    pub(crate) fn switch() {
        use std::future::Future;
        use std::pin::Pin;
        use std::ptr;
        use std::task::{Context, RawWaker, RawWakerVTable, Waker};

        unsafe fn noop_clone(_: *const ()) -> RawWaker {
            unreachable!()
        }
        unsafe fn noop(_: *const ()) {}

        // Wrapping with an async block deals with the thread-local context
        // `std` uses to manage async blocks
        let mut switch = async { generator::yield_with(()) };
        let switch = unsafe { Pin::new_unchecked(&mut switch) };

        let raw_waker = RawWaker::new(
            ptr::null(),
            &RawWakerVTable::new(noop_clone, noop, noop, noop),
        );
        let waker = unsafe { Waker::from_raw(raw_waker) };
        let mut cx = Context::from_waker(&waker);

        assert!(switch.poll(&mut cx).is_ready());
    }

    pub(crate) fn spawn(stack_size: Option<usize>, f: Box<dyn FnOnce()>) {
        Self::with_state(|state| state.queued_spawn.push_back(QueuedSpawn { stack_size, f }));
    }

    // 调度器运行模型内闭包, Execution管理线程生命周期
    pub(crate) fn run<F>(&mut self, execution: &mut Execution, f: F)
    where
        F: FnOnce() + Send + 'static,
    {
        let mut threads = Vec::new();
        threads.push(spawn_thread(Box::new(f), None));
        // threads[0].resume()启动容器中的第一个（也是唯一的）线程,即Model中闭包
        threads[0].resume();

        loop {
            // 这个循环是核心的调度器部分，它不断检查执行状态并管理线程。
            // if execution.threads.is_complete()检查所有线程是否已经完成
            // 在所有线程完成后，对每个线程调用resume()以确保它们都已经结束
            // 然后通过assert!(thread.is_done())确认这一点
            if execution.threads.is_complete() {
                for thread in &mut threads {
                    thread.resume();
                    assert!(thread.is_done());
                }
                return;
            }

            let active = execution.threads.active_id();

            let mut queued_spawn = Self::tick(&mut threads[active.as_usize()], execution);

            while let Some(th) = queued_spawn.pop_front() {
                assert!(threads.len() < self.max_threads);

                let thread_id = threads.len();
                let QueuedSpawn { f, stack_size } = th;

                threads.push(spawn_thread(f, stack_size));
                threads[thread_id].resume();
            }
        }
    }

    fn tick(thread: &mut Thread, execution: &mut Execution) -> VecDeque<QueuedSpawn> {
        let mut queued_spawn = VecDeque::new();
        let state = RefCell::new(State {
            execution,
            queued_spawn: &mut queued_spawn,
        });
        // 临时设置线程局部状态
        STATE.set(unsafe { transmute_lt(&state) }, || {
            // 恢复线程的执行，期间可能通过STATE访问execution和queued_spawn
            thread.resume();
        });
        // 返回此次tick调用期间收集的所有新线程生成请求
        queued_spawn
    }

    fn with_state<F, R>(f: F) -> R
    where
        F: FnOnce(&mut State<'_>) -> R,
    {
        if !STATE.is_set() {
            panic!("cannot access Loom execution state from outside a Loom model. \
            are you accessing a Loom synchronization primitive from outside a Loom test (a call to `model` or `check`)?")
        }
        STATE.with(|state| f(&mut state.borrow_mut()))
    }
}

// FnOnce转移自由变量所有权
fn spawn_thread(f: Box<dyn FnOnce()>, stack_size: Option<usize>) -> Thread {
    // body是一个generator, 无限次产生值的生成器,用于产生无限序列
    let body = move || {
        loop {
            // yield_产生值并可能接收外部通过set_para传入的新闭包,首次传入的是f
            let f: Option<Option<Box<dyn FnOnce()>>> = generator::yield_(());

            if let Some(f) = f {
                // 再次暂停
                generator::yield_with(());
                // 执行此闭包
                f.unwrap()();
            } else {
                break;
            }
        }

        generator::done!();
    };
    let mut g = match stack_size {
        Some(stack_size) => Gn::new_opt(stack_size, body),
        None => Gn::new(body),
    };
    g.resume();
    g.set_para(Some(f));
    g
}

unsafe fn transmute_lt<'a, 'b>(state: &'a RefCell<State<'b>>) -> &'a RefCell<State<'static>> {
    ::std::mem::transmute(state)
}
