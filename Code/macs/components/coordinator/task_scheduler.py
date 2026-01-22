from collections import defaultdict, deque
from typing import cast
# Local imports
from macs.components.domain.schema import MultiColumnTask, FDResult

class TaskScheduler:
    """Determines the correct execution order of functional dependencies based on their dependency relationships."""
    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def get_execution_order(self, multi_col_tasks: list[MultiColumnTask]) -> list[list[MultiColumnTask]]:
        """
        Determines the correct cleaning order for dependent tasks using topological sort to respect dependencies.
        """        
        if not multi_col_tasks:
            return []
        # 1. Seperate FDs from other constraints. FDs have strict dependencies, others might not
        fd_tasks = []
        other_tasks = []
        for task in multi_col_tasks:
            if task.task_type == 'FD':
                fd_tasks.append(task)
            else:
                other_tasks.append(task)
        # 2. Map FD tasks to indices for topological sort      
        fds_map = {i: fd for i, fd in enumerate(fd_tasks)}
        current_fd_ids = set(fds_map.keys())
        final_ordered_level_ids = []
        # 3. Run topologic sort
        while True:
            if not current_fd_ids:
                break
            ordered_levels_ids = self._run_topological_sort(fds_map, current_fd_ids)
            final_ordered_level_ids.extend(ordered_levels_ids)
            processed_ids = {fd_idx for level in ordered_levels_ids for fd_idx in level}
            # If not all FDs are processed, there is a cycle that needs to be broken
            current_fd_ids -= processed_ids
            if not current_fd_ids:
                break
            cycle_fd_ids = self._find_actual_cycle(fds_map, current_fd_ids)
            if not cycle_fd_ids:
                cycle_fd_ids = current_fd_ids
            weakest_fd_idx = min(
                cycle_fd_ids, 
                key=lambda fd_idx: cast(FDResult, fds_map[fd_idx].data).score
            )
            current_fd_ids.remove(weakest_fd_idx)
        # 4. Define final ordered list
        final_ordered_levels = []
        for level_ids in final_ordered_level_ids:
            level_fds = [fds_map[fd_idx] for fd_idx in level_ids]
            final_ordered_levels.append(level_fds)
        if other_tasks:
            final_ordered_levels.append(other_tasks)        
        return final_ordered_levels

    @staticmethod
    def _get_lhs_rhs(task: MultiColumnTask) -> tuple[str,str]:
        data = cast(FDResult, task.data)
        return data.lhs, data.rhs

    def _find_actual_cycle(
            self,
            fds_map: dict[int, dict],
            stuck_fd_ids: set[int]) -> set[int]:
        """
        Finds the FDs that are actually in a cycle from a list of all stuck FDs.
        Performs reverse topological sort (pruning leaf nodes) until only cycle remains.
        """
        candidate_fds = set(stuck_fd_ids) 
        while True:
            stuck_lhs_set = {self._get_lhs_rhs(fds_map[fd_id])[0] for fd_id in candidate_fds}
            # Find FD's RHS that do not point to another stuck FD (no LHS somewhere). These are not part of cycle
            downstream_fds = set()
            for idx in candidate_fds:
                _, rhs = self._get_lhs_rhs(fds_map[idx])
                if rhs not in stuck_lhs_set:
                    downstream_fds.add(idx)
            if not downstream_fds:
                return candidate_fds
            candidate_fds -= downstream_fds

    def _run_topological_sort(
            self,
            fds_map: dict[int, dict],
            current_fd_ids: set[int]) -> tuple[list[list[int]], set[int]]:
        """Determines which FDs can be enforced simultaneously, respecting the dependencies."""
        # 1. Initialise FD 'graph'
        rhs_fd_map = defaultdict(list)
        in_degree = defaultdict(int)
        for idx in current_fd_ids:
            in_degree[idx] = 0 
            _, rhs = self._get_lhs_rhs(fds_map[idx])
            rhs_fd_map[rhs].append(idx)
        # 2. Identify dependent FDs
        for idx_j in current_fd_ids:
            lhs, _ = self._get_lhs_rhs(fds_map[idx_j])
            dependent_on_fds = rhs_fd_map.get(lhs, [])
            count = sum(1 for dep_idx in dependent_on_fds if dep_idx != idx_j)
            in_degree[idx_j] = count
        # 3. Add FDs without dependency to queue
        queue = deque([idx for idx in current_fd_ids if in_degree[idx] == 0])
        ordered_levels_ids = []
        ordered_fd_ids = set()
        # 4. Solve dependencies by sorting FDs in levels
        while queue:
            level_ids = []
            for _ in range(len(queue)):
                fd_idx = queue.popleft()
                if fd_idx in ordered_fd_ids: 
                    continue                 
                level_ids.append(fd_idx)
                ordered_fd_ids.add(fd_idx)
                _, cleaned_col = self._get_lhs_rhs(fds_map[fd_idx])
                for dep_idx in current_fd_ids:
                    if dep_idx not in ordered_fd_ids:
                        dep_lhs, _ = self._get_lhs_rhs(fds_map[dep_idx])
                        if dep_lhs == cleaned_col:
                            in_degree[dep_idx] -= 1
                            if in_degree[dep_idx] == 0:
                                queue.append(dep_idx)
            if level_ids:
                ordered_levels_ids.append(level_ids)
        return ordered_levels_ids
            
####### TEST CODE #######
if __name__ == "__main__":
    hospital_fds = [
        {'lhs': 'Address1' , 'rhs': 'HospitalName' , 'score': 0.9770}, 
        {'lhs': 'PhoneNumber' , 'rhs': 'State', 'score': 0.9760 },
        {'lhs': 'PhoneNumber' , 'rhs': 'HospitalOwner', 'score': 0.9750  },
        {'lhs': 'Address1' , 'rhs': 'EmergencyService', 'score': 0.9740  },
        {'lhs': 'Address1' , 'rhs': 'ProviderNumber', 'score': 0.9720  },
        {'lhs': 'Stateavg' , 'rhs': 'MeasureCode', 'score': 0.9720  },
        {'lhs': 'PhoneNumber' , 'rhs': 'City', 'score': 0.9690  }, 
        {'lhs': 'Stateavg' , 'rhs': 'Condition', 'score': 0.9690  },   
        {'lhs': 'Stateavg' , 'rhs': 'MeasureName', 'score': 0.9660  },  
        {'lhs': 'PhoneNumber' , 'rhs': 'Address1', 'score': 0.9700 }, 
        {'lhs': 'City' , 'rhs': 'CountyName', 'score': 0.9640  },
        {'lhs': 'MeasureName' , 'rhs': 'Stateavg', 'score': 0.9550  }, 
        {'lhs': 'HospitalName' , 'rhs': 'ZipCode', 'score': 0.9710  },
        {'lhs': 'Address1' , 'rhs': 'PhoneNumber', 'score': 0.9680 }
    ]

    multi_col_tasks = [
        MultiColumnTask(
            task_type="FD",
            target_columns=[item['lhs'], item['rhs']],
            verbose_key=f"{item['lhs']} -> {item['rhs']}",
            data=FDResult(
                lhs=item['lhs'],
                rhs=item['rhs'],
                score=item['score']
            ))
        for item in hospital_fds
    ]
    multi_col_tasks.append(MultiColumnTask(
        task_type="A",
        target_columns=['B', 'C'],
        verbose_key="B <= C",
        data=5
    ))
    task_scheduler = TaskScheduler(verbose=True)
    fd_groups = task_scheduler.get_execution_order(multi_col_tasks)
    print(fd_groups[0])
    print()
    print(fd_groups[-1])

    # python -m src.components.coordinator.task_scheduler





