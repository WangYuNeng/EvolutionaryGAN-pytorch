
class MinHeap:
    def __init__(self, array=()):
        # array must be heapified first, which is true in egan_model
        self.array = array

    def push(self, G_Net):
        '''
        push an item to the heap
        G_Net(fitness, G_candis, optG_candis)
        '''
        cur_idx = len(self.array)
        self.array.append(None)
        while cur_idx != 0:
            parent_idx = (cur_idx-1) // 2
            if self.array[parent_idx].fitness <= G_Net.fitness:
                break
            else:
                self.array[cur_idx] = self.array[parent_idx]
                cur_idx = parent_idx
        self.array[cur_idx] = G_Net

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

    def replace(self, G_Net):
        '''
        replace the smallest item with input item
        '''
        self.array[0] = G_Net
        self._heapify_top()

    def argmax(self):
        # find argmax fitness netG
        max_val, max_idx = -float('inf'), 0
        for i, net in enumerate(self.array):
            if net.fitness > max_val:
                max_val, max_idx = net.fitness, i
        return max_idx

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
                if self.array[cur_idx].fitness > self.array[left_idx].fitness:
                    self._swap(self.array, cur_idx, left_idx)
                break
            else:
                smaller_idx = left_idx if self.array[right_idx].fitness > self.array[left_idx].fitness else right_idx
                if self.array[cur_idx].fitness > self.array[smaller_idx].fitness:
                    self._swap(self.array, cur_idx, smaller_idx)
                    cur_idx = smaller_idx
                else:
                    break
    @staticmethod
    def _swap(array, idx1, idx2):
        array[idx1], array[idx2] = array[idx2], array[idx1]

