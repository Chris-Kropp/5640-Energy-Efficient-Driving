import carla
import random
import time
import os

import networkx as nx
import osmnx as ox
import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, mapping
from energy_model.models.fastsim_model import fastsim_energy_model
from energy_model.models.simple_energy_model import ev_energy_model
import math
from multiprocessing import Process, Value
import torch

outputs = [(0.1, 0), (0.2, 0), (0.3, 0), (0.4, 0), (0.5, 0), (0.6, 0), (0.7, 0), (0.8, 0), (0.9, 0), (0, 0), (1, 0), (0, 0.1), (0, 0.2), (0, 0.3), (0, 0.4), (0, 0.5), (0, 0.6), (0, 0.7), (0, 0.8), (0, 0.9), (0, 1)]

agentType = "roadVisible" # "instantaneous" or "roadVisible"

firstMid = -164.82841
secondMid = -42.00291
height = 7.99823

leftSlope = 1.18750
rightSlope = 1.10655

def getHeight(x):
    return ((height)/(leftSlope**(firstMid-x)+1)) - ((height)/(rightSlope**(secondMid-x)+1))

def getSlope(x):
    return (((np.log(leftSlope)*height)*leftSlope**(-x+firstMid))/(((leftSlope**(firstMid-x))+1)**2) - ((np.log(rightSlope)*height)*rightSlope**(-x+secondMid))/(((rightSlope**(secondMid-x))+1)**2))


os.system("kill `pidof CarlaUE4-Linux-Shipping`")

try:
    os.remove("reward.txt")
    os.remove("distance.txt")
    os.remove("energy.txt")
except:
    print("could not delete, check correctness")

energyModel = ev_energy_model(1580, 0.19, 2.55)

def startCarla():
    os.system("bash /home/chris/Downloads/CARLA_0.9.13/CarlaUE4.sh")

if(agentType == "instantaneous"):
    class NeuralNetwork(torch.nn.Module):
        def __init__(self):
            super(NeuralNetwork, self).__init__()
            self.inputLayer = torch.nn.Linear(7, 14)
            self.hiddenLayer1 = torch.nn.Linear(14, 56)
            self.hiddenLayer2 = torch.nn.Linear(56, 50)
            self.outputLayer = torch.nn.Linear(50, 21)

        def forward(self, state):
            state = self.inputLayer(state)
            state = self.hiddenLayer1(state)
            state = self.hiddenLayer2(state)
            state = self.outputLayer(state)
            state = torch.nn.functional.relu(state)
            return state

    class Agent:
        def __init__(self):
            self.device = torch.device("cpu")
            self.model = NeuralNetwork().to(self.device)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-3)

            self.decay = 0.995
            self.randomness = 0.00
            self.min_randomness = 0.001

        def act(self, state):
            if(self.randomness > self.min_randomness):
                self.randomness *= self.decay
            state = torch.from_numpy(state).unsqueeze(0).to(self.device)
            qualities = self.model(state).cpu()
            probs = torch.distributions.Categorical(qualities)
            action = probs.sample()
            if(random.random() > self.randomness):
                return outputs[action], probs.log_prob(action)
            else:
                return outputs[random.randint(0, len(outputs)-1)], probs.log_prob(action)


        def updateReward(self, energy=0, distance=0):
            self.energy += energy
            self.distance += distance

        def update(self, ):
            self.randomness *= self.decay
            self.randomness = max(self.randomness, self.min_randomness)

elif(agentType == "roadVisible"):
    class NeuralNetwork(torch.nn.Module):
        def __init__(self):
            super(NeuralNetwork, self).__init__()
            self.inputLayer = torch.nn.Linear(17, 24)
            self.hiddenLayer1 = torch.nn.Linear(24, 56)
            self.hiddenLayer2 = torch.nn.Linear(56, 50)
            self.outputLayer = torch.nn.Linear(50, 21)

        def forward(self, state):
            state = self.inputLayer(state)
            state = self.hiddenLayer1(state)
            state = self.hiddenLayer2(state)
            state = self.outputLayer(state)
            state = torch.nn.functional.relu(state)
            return state

    class Agent:
        def __init__(self):
            self.device = torch.device("cpu")
            self.model = NeuralNetwork().to(self.device)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-3)

            self.decay = 0.995
            self.randomness = 0.00
            self.min_randomness = 0.001

        def act(self, state):
            if(self.randomness > self.min_randomness):
                self.randomness *= self.decay
            state = torch.from_numpy(state).unsqueeze(0).to(self.device)
            qualities = self.model(state).cpu()
            probs = torch.distributions.Categorical(qualities)
            action = probs.sample()
            a = probs.log_prob(action)
            if(random.random() > self.randomness):
                return outputs[action], probs.log_prob(action)
            else:
                return outputs[random.randint(0, len(outputs)-1)], probs.log_prob(action)


        def updateReward(self, energy=0, distance=0):
            self.energy += energy
            self.distance += distance

        def update(self, ):
            self.randomness *= self.decay
            self.randomness = max(self.randomness, self.min_randomness)

agent = Agent()

def computeReward(energy, distance, velocity):
    if(distance > 205):
        # return (.0000001*(-1*energy)) + (10*distance) - velocity**2
        return (1300*(-1*energy)) + (10*distance) - velocity**2
    # return (.0000001*(-1*energy)) + (10*distance)
    return (1300*(-1*energy)) + (10*distance)


while(True):
    try:
        carlaProcess = Process(target=startCarla)
        carlaProcess.start()
        time.sleep(5)

        try:
            client = carla.Client('localhost', 2000)
            client.set_timeout(2.0)

            world = client.get_world()
        except:
            time.sleep(10)
            client = carla.Client('localhost', 2000)
            client.set_timeout(2.0)

            world = client.get_world()


        blueprint_library = world.get_blueprint_library()
        bp = random.choice(blueprint_library.filter('micra'))
        for actor in world.get_actors():
            if(actor.type_id != "spectator"):
                actor.destroy()

        transform = carla.Transform(carla.Location(x=79.4, y=-215, z=0), carla.Rotation(yaw=91.5))

        frame = world.apply_settings(carla.WorldSettings(
            synchronous_mode=True,
            fixed_delta_seconds=1/15))

        count = 0
        spectator =  world.get_spectator()

        oldY, oldZ = -215, 0

        step_count = 0
        avg_grade = 0
        grade = 0
        speed = 0

        vehicle = world.spawn_actor(bp, transform)

        world.tick()

        world = client.get_world()

        loc = -215

        itercount = 0
        totalDistance = 0
        totalEnergy = 0
        probs = []

        # accel = []
        # brake = []
        try:
            os.remove("accel.txt")
            os.remove("brake.txt")
        except:
            print("could not delete, check correctness")
        while(count < 400 and loc < -5):
            location = vehicle.get_transform().location
            loc = location.y
            
            observedSpeed = vehicle.get_velocity()
            observedAcceleration = vehicle.get_acceleration()
            if(agentType == "instantaneous"):
                observation = np.array([observedSpeed.x, observedSpeed.y, observedSpeed.z, grade, observedAcceleration.x, observedAcceleration.y, observedAcceleration.z], dtype=np.float32)
            elif(agentType == "roadVisible"):
                observation = np.array([observedSpeed.x, observedSpeed.y, observedSpeed.z, grade, observedAcceleration.x, observedAcceleration.y, observedAcceleration.z, getHeight(loc+1), getHeight(loc+3), getHeight(loc+5), getHeight(loc+10), getHeight(loc+20), getSlope(loc+1), getSlope(loc+3), getSlope(loc+5), getSlope(loc+10), getSlope(loc+20)], dtype=np.float32)
            control, prob = agent.act(observation)
            vehicle.apply_control(carla.VehicleControl(throttle=control[0], steer=0.0, brake=control[1], hand_brake=False, reverse=False, manual_gear_shift=False, gear=1))
            
            probs.append(prob)
            
            world.tick()
            
            spectator.set_transform(carla.Transform(carla.Location(x=location.x+20, y=location.y, z=30), carla.Rotation(pitch=-45, yaw=180, roll=0)))
            
            length = location.y-oldY
            if(length > 0):
                grade = math.tan((location.z-oldZ)/length)
                grade *= math.pi/180
            else:
                grade = 0
            speed_mps = vehicle.get_velocity().y
            avg_grade += grade
            avg_grade /= 2
            speed += speed_mps
            speed /= 2
            step_count += 1
            count += 1
            
            if(step_count == 10):
                # energy_consumption = energyModel.energy_consumption(0, speed, control[0], 0, 0, 0, length)
                energy_consumption = energyModel.get_consumed_kwh_fastsim(speed, avg_grade, length, 0)
                totalEnergy += energy_consumption
                totalDistance += length
                itercount += 1
                step_count = 0
                avg_grade = 0
                speed = 0
                length = 0
                oldY = location.y
                oldZ = location.z
            itercount += 1

            with open("accel.txt", 'a') as outfile:
                outfile.write(str(control[0]) + '\n')
            with open("brake.txt", 'a') as outfile:
                outfile.write(str(control[1]) + '\n')
        
        reward = computeReward(totalEnergy, totalDistance, vehicle.get_velocity().y)
        policy_loss = [-log_prob * reward for log_prob in probs]
        policy_loss = torch.cat(policy_loss).sum()
        agent.optimizer.zero_grad()
        policy_loss.backward()
        agent.optimizer.step()

        with open("reward.txt", 'a') as outfile:
            outfile.write(str(reward) + '\n')
        with open("energy.txt", 'a') as outfile:
            outfile.write(str(totalEnergy) + '\n')
        with open("distance.txt", 'a') as outfile:
            outfile.write(str(totalDistance) + '\n')

        os.system("kill `pidof CarlaUE4-Linux-Shipping`")
        try:
            world.tick()
        except:
            os.system("kill `pidof CarlaUE4-Linux-Shipping`")
        carlaProcess.terminate()

        time.sleep(1)

    except Exception as e:
        os.system("kill `pidof CarlaUE4-Linux-Shipping`")
