
class minheap:
    def __init__(self, array=list()):
        # array must be heapified first, which is true in egan_model
        self.array = array

    def push(self, item):
        '''
        push an item to the heap
        item <- (key, value)
        '''
        cur_idx = len(self.array)
        self.array.append(None)
        while cur_idx != 0:
            parent_idx = (cur_idx-1) // 2
            if self.array[parent_idx][0] <= item[0]:
                break
            else:
                self.array[cur_idx] = self.array[parent_idx]
                cur_idx = parent_idx
        self.array[cur_idx] = item

    def pop(self):
        '''
        return the smallest item in the heap
        '''
        r_value = self.array[0]
        self.array[0] = self.array[-1]
        self.array = self.array[:-1]

        self._heapify_top()
        return r_value

    def top(self):
        '''
        return the top(smallest) item in the heap
        '''
        return self.array[0]

    def replace(self, item):
        '''
        replace the smallest item with input item
        '''
        self.array[0] = item
        self._heapify_top()

    def _heapify_top(self):
        '''
        maintain the heap after the top item is changed
        '''
        cur_idx = 0
        while True:
            left_idx = 2*cur_idx + 1
            right_idx = 2*cur_idx + 2

            if left_idx >= len(self.array):
                break
            elif right_idx >= len(self.array):
                if self.array[cur_idx][0] > self.array[left_idx][0]:
                    self._swap(cur_idx, left_idx)
                break
            else:
                smaller_idx = left_idx if self.array[right_idx][0] > self.array[left_idx][0] else right_idx
                if self.array[cur_idx][0] > self.array[smaller_idx][0]:
                    self._swap(cur_idx, smaller_idx)
                    cur_idx = smaller_idx
                else:
                    break

    def _swap(self, idx1, idx2):
        tmp = self.array[idx1]
        self.array[idx1] = self.array[idx2]
        self.array[idx2] = tmp

