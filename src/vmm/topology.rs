/// CPU topology specification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Topology {
    pub sockets: u32,
    pub cores_per_socket: u32,
    pub threads_per_core: u32,
}

impl Topology {
    pub fn total_cpus(&self) -> u32 {
        self.sockets * self.cores_per_socket * self.threads_per_core
    }

    pub fn num_llcs(&self) -> u32 {
        self.sockets
    }

    /// Decompose a logical CPU ID into (socket, core, thread).
    pub fn decompose(&self, cpu_id: u32) -> (u32, u32, u32) {
        let threads = self.threads_per_core;
        let cores = self.cores_per_socket;
        let thread_id = cpu_id % threads;
        let core_id = (cpu_id / threads) % cores;
        let socket_id = cpu_id / (threads * cores);
        (socket_id, core_id, thread_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn topology_total_cpus() {
        let t = Topology {
            sockets: 2,
            cores_per_socket: 4,
            threads_per_core: 2,
        };
        assert_eq!(t.total_cpus(), 16);
    }

    #[test]
    fn topology_num_llcs() {
        let t = Topology {
            sockets: 3,
            cores_per_socket: 4,
            threads_per_core: 2,
        };
        assert_eq!(t.num_llcs(), 3);
    }

    #[test]
    fn decompose_simple() {
        let t = Topology {
            sockets: 2,
            cores_per_socket: 2,
            threads_per_core: 2,
        };
        // 2 sockets x 2 cores x 2 threads = 8 CPUs
        assert_eq!(t.decompose(0), (0, 0, 0));
        assert_eq!(t.decompose(1), (0, 0, 1));
        assert_eq!(t.decompose(2), (0, 1, 0));
        assert_eq!(t.decompose(3), (0, 1, 1));
        assert_eq!(t.decompose(4), (1, 0, 0));
        assert_eq!(t.decompose(5), (1, 0, 1));
        assert_eq!(t.decompose(6), (1, 1, 0));
        assert_eq!(t.decompose(7), (1, 1, 1));
    }

    #[test]
    fn decompose_no_smt() {
        let t = Topology {
            sockets: 2,
            cores_per_socket: 4,
            threads_per_core: 1,
        };
        assert_eq!(t.decompose(0), (0, 0, 0));
        assert_eq!(t.decompose(3), (0, 3, 0));
        assert_eq!(t.decompose(4), (1, 0, 0));
        assert_eq!(t.decompose(7), (1, 3, 0));
    }

    #[test]
    fn decompose_single_socket() {
        let t = Topology {
            sockets: 1,
            cores_per_socket: 4,
            threads_per_core: 1,
        };
        assert_eq!(t.decompose(0), (0, 0, 0));
        assert_eq!(t.decompose(3), (0, 3, 0));
    }
}
