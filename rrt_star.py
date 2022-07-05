import time
import math
import tqdm
import numpy as np

from tool import env, plotting, utils, Node


class RrtStar:
    def __init__(self, start_point, end_point, step_length, search_radius, sample_rate, max_iter, env_instance):
        '''
        :param start_point: start point of the robot
        :param end_point: end point of the robot
        :param step_length: step length of the robot
        :param search_radius: frnn radius
        :param sample_rate: sample rate for finding new node
        :param max_iter: maximum iteration
        :param env_instance:
        '''
        self.st_point = Node(start_point)
        self.ed_point = Node(end_point)
        self.step_length = step_length
        self.search_radius = search_radius
        self.sample_rate = sample_rate
        self.max_iter = max_iter
        self.nodes = [self.st_point]

        # initialize the environment
        self.env = env_instance
        self.plotting = plotting.Plotting(start_point, end_point, self.env)
        self.utils = utils.Utils(self.env)
        self.utils.fix_random_seed()
        self.x_range = self.env.x_range
        self.y_range = self.env.y_range
        self.obs_circle = self.env.obs_circle
        self.obs_rectangle = self.env.obs_rectangle
        self.obs_boundary = self.env.obs_boundary

        # record
        self.iter_num = -1
        self.time_start = -1
        self.time_end = -1
        self.dist = -1
        
    def generate_random_node(self, goal_sample_rate):
        delta = self.utils.delta # 0.5
        
        if np.random.random() > goal_sample_rate:
            return Node((np.random.uniform(self.x_range[0] + delta, self.x_range[1] - delta),
                         np.random.uniform(self.y_range[0] + delta, self.y_range[1] - delta)))
            
        return self.ed_point
    
    def nearest_neighbor(self, node_list, n):
        return node_list[int(np.argmin([math.hypot(nd.x - n.x, nd.y - n.y)
                                        for nd in node_list]))]
    
    def new_state(self, node_start, node_goal):
        dist, theta = self.get_distance_and_angle(node_start, node_goal)
        
        dist = min(self.step_length, dist)
        node_new = Node((node_start.x + dist * math.cos(theta),
                         node_start.y + dist * math.sin(theta)))
        node_new.parent = node_start
        
        return node_new
    
    def get_distance_and_angle(self, node_start, node_end):
        dx = node_end.x - node_start.x
        dy = node_end.y - node_start.y
        return math.hypot(dx,dy), math.atan2(dy, dx)
    
    def find_near_neighbor(self, node_new):
        n = len(self.nodes) + 1
        r = min(self.search_radius * math.sqrt((math.log(n) / n)), self.step_length)
        dist_table = [math.hypot(nd.x - node_new.x, nd.y - node_new.y) for nd in self.nodes]
        dist_table_index = [ind for ind in range(len(dist_table)) if dist_table[ind] <= r and
                             not self.utils.is_collision(node_new, self.nodes[ind])]
        
        return dist_table_index
    
    def cost(self,node_p):
        node = node_p
        cost = 0.0
        while node.parent:
            cost += math.hypot(node.x - node.parent.x, node.y - node.parent.y)
            node = node.parent
        
        return cost
    
    def choose_parent(self, node_new, neighbor_index):
        cost = [self.get_new_cost(self.nodes[i], node_new) for i in neighbor_index]
        cost_min_index = neighbor_index[int(np.argmin(cost))]
        node_new.parent = self.nodes[cost_min_index]
        
    def get_new_cost(self, node_start, node_end):
        dist, _ = self.get_distance_and_angle(node_start, node_end)
        
        return self.cost(node_start)  + dist
     
    def rewire(self, node_new, neighbor_index):
        for i in neighbor_index:
            node_neighbor = self.nodes[i]
            if self.cost(node_neighbor) > self.get_new_cost(node_new,node_neighbor):
                node_neighbor.parent = node_new
                
    def search_goal_parent(self):
        dist_list = [math.hypot(n.x - self.ed_point.x, n.y - self.ed_point.y) for n in self.nodes]
        node_index = [i for i in range(len(dist_list)) if dist_list[i]<= self.step_length]
        
        if len(node_index) > 0:
            cost_list = [dist_list[i] + self.cost(self.nodes[i]) for i in node_index
                         if not self.utils.is_collision(self.nodes[i], self.ed_point)]
            return node_index[int(np.argmin(cost_list))]
        return len(self.vertex) - 1
    
    def planning(self):
        path = None
        self.time_start = time.time()
        converge = False
        for i in tqdm.tqdm(range(self.max_iter)):
            # TODO: Implement RRT Star planning (Free to add your own functions)
            # 在一定概率选目标点的同时随机采点
            node_rand = self.generate_random_node(self.sample_rate)
            node_near = self.nearest_neighbor(self.nodes, node_rand)
            node_new = self.new_state(node_near, node_rand)
            
            if i % 500 == 0:
                print(i)
                
            if node_new and not self.utils.is_collision(node_near, node_new):
                neighbor_index = self.find_near_neighbor(node_new)
                self.nodes.append(node_new)

                if neighbor_index:
                    self.choose_parent(node_new, neighbor_index)
                    self.rewire(node_new, neighbor_index)
                    
        index = self.search_goal_parent()
        
        self.path = self.extract_path(self.nodes[index])
        
        # self.plotting.animation(self.nodes, self.path, "rrt*, N = " + str(self.max_iter)) 
                 
        return self.path
                 

        self.time_end = time.time()
        self.iter_num = i + 1
        # return final path
        # implement extract_path func maybe help
        # path = extract_path(...)
        # self.dist = self.path_distance(path)
        return path

    def extract_path(self, node_end):
        '''
        extract the path from the end node by backtracking
        :param node_end:
        :return:
        '''
        path = [(self.ed_point.x, self.ed_point.y)]
        node_now = node_end

        while node_now.parent is not None:
            node_now = node_now.parent
            path.append((node_now.x, node_now.y))

        return path

    def path_distance(self, path):
        '''
        get the distance of the path
        :param path:
        :return:
        '''
        dist_sum = 0
        for i in range(len(path) - 1):
            dist_sum += np.linalg.norm([path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1]])
        return dist_sum


def env1_planning(eval_time=1):
    x_start = (5, 5)  # st node
    x_goal = (49, 16)  # end node

    # visualization
    if eval_time == 1:
        rrt = RrtStar(x_start, x_goal, 1, 0.05, 0.1, 10000, env.EnvOne())
        path = rrt.planning()
        if not path:
            print("No Path Found!")
        print("Time: {:.3f} s".format(rrt.time_end - rrt.time_start))
        print("Iteration:", rrt.iter_num)
        if path:
            print("Distance:{:.3f}".format(rrt.dist))
        rrt.plotting.animation(rrt.nodes, path, "RRT_STAR_ENV1", True)
        return
    # evaluation
    time_sum = list()
    iter_sum = list()
    dist_sum = list()
    for i in range(eval_time):
        rrt = RrtStar(x_start, x_goal, 0.5, 0.05, 0.1, 10000, env.EnvOne())
        path = rrt.planning()
        if not path:
            print("No Path Found!")
        print("###### Evaluation: {} ######".format(i + 1))
        print("Time: {:.3f} s".format(rrt.time_end - rrt.time_start))
        print("Iteration:", rrt.iter_num)
        if path:
            print("Distance:{:.3f}".format(rrt.dist))
            dist_sum.append(rrt.dist)
        print("-----------------------------------------------------")
        time_sum.append(rrt.time_end - rrt.time_start)
        iter_sum.append(rrt.iter_num)

    # average time
    print("Average Time: {:.3f} s".format(np.mean(time_sum)))
    # average iteration
    print("Average Iteration: {:.0f}".format(np.mean(iter_sum)))
    # average distance
    if len(dist_sum) > 0:
        print("Average Distance: {:.3f}".format(np.mean(dist_sum)))


def env2_planning(eval_time=1):
    x_start = (5, 20)  # st node
    x_goal = (67, 40)  # end node

    # visualization
    if eval_time == 1:
        rrt = RrtStar(x_start, x_goal, 0.5, 0.2, 0.1, 10000, env.EnvTwo())
        path = rrt.planning()
        if not path:
            print("No Path Found!")
        print("Time: {:.3f} s".format(rrt.time_end - rrt.time_start))
        print("Iteration:", rrt.iter_num)
        if path:
            print("Distance:{:.3f}".format(rrt.dist))
        rrt.plotting.animation(rrt.nodes, path, "RRT_STAR_ENV2", True)
        return
    # evaluation
    time_sum = list()
    iter_sum = list()
    dist_sum = list()
    for i in range(eval_time):
        rrt = RrtStar(x_start, x_goal, 0.5, 0.2, 0.1, 10000, env.EnvTwo())
        path = rrt.planning()
        if not path:
            print("No Path Found!")
        print("###### Evaluation: {} ######".format(i + 1))
        print("Time: {:.3f} s".format(rrt.time_end - rrt.time_start))
        print("Iteration:", rrt.iter_num)
        if path:
            print("Distance:{:.3f}".format(rrt.dist))
            dist_sum.append(rrt.dist)
        print("-----------------------------------------------------")
        time_sum.append(rrt.time_end - rrt.time_start)
        iter_sum.append(rrt.iter_num)

    # average time
    print("Average Time: {:.3f} s".format(np.mean(time_sum)))
    # average iteration
    print("Average Iteration: {:.0f}".format(np.mean(iter_sum)))
    # average distance
    if len(dist_sum) > 0:
        print("Average Distance: {:.3f}".format(np.mean(dist_sum)))


def env3_planning(eval_time=1):
    x_start = (5, 2)  # st node
    x_goal = (18, 18)  # end node

    # visualization
    if eval_time == 1:
        rrt = RrtStar(x_start, x_goal, 0.5, 0.1, 0.2, 10000, env.EnvThree())
        path = rrt.planning()
        if not path:
            print("No Path Found!")
        print("Time: {:.3f} s".format(rrt.time_end - rrt.time_start))
        print("Iteration:", rrt.iter_num)
        print("Distance:{:.3f}".format(rrt.dist))
        rrt.plotting.animation(rrt.nodes, path, "RRT_STAR_ENV3", True)
        return
    # evaluation
    time_sum = list()
    iter_sum = list()
    dist_sum = list()
    for i in range(eval_time):
        rrt = RrtStar(x_start, x_goal, 0.5, 0.2, 0.1, 10000, env.EnvThree())
        path = rrt.planning()
        if not path:
            print("No Path Found!")
        print("###### Evaluation: {} ######".format(i + 1))
        print("Time: {:.3f} s".format(rrt.time_end - rrt.time_start))
        print("Iteration:", rrt.iter_num)
        if path:
            print("Distance:{:.3f}".format(rrt.dist))
            dist_sum.append(rrt.dist)
        print("-----------------------------------------------------")
        time_sum.append(rrt.time_end - rrt.time_start)
        iter_sum.append(rrt.iter_num)

    # average time
    print("Average Time: {:.3f} s".format(np.mean(time_sum)))
    # average iteration
    print("Average Iteration: {:.0f}".format(np.mean(iter_sum)))
    # average distance
    if len(dist_sum) > 0:
        print("Average Distance: {:.3f}".format(np.mean(dist_sum)))


if __name__ == '__main__':
    env1_planning(eval_time=1)
    # env2_planning(eval_time=1)
    # env3_planning(eval_time=1)
