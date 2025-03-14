from math import tau, sqrt, atan2, floor
from os import path, mkdir, rmdir, remove
from glob import glob
from time import sleep
import cairo
import ffmpeg
import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import pyinputplus as pyip
from colorama import Fore, Back, Style
from softquake import RigidPlate, Sine, Load, Sensor
from softbodies import Node, Link
from vectors import Vector

draw_text = True

try:
    colors = pyip.inputMenu(["Yes", "No"], prompt="Show colored text:\n", lettered=True)

    if colors == "Yes":
        colors = True
    elif colors == "No":
        colors = False

    if colors is not True:
        class Fore:
            WHITE = ""
            RED = ""
            BLUE = ""
            BLACK = ""
            YELLOW = ""
            MAGENTA = ""
            LIGHTWHITE_EX = ""
            LIGHTRED_EX = ""
            LIGHTBLUE_EX = ""
            LIGHTBLACK_EX = ""
            LIGHTGREEN_EX = ""
            LIGHTMAGENTA_EX = ""
            RESET = ""


        class Back:
            BLACK = ""
            BLUE = ""
            RED = ""
            LIGHTYELLOW_EX = ""
            LIGHTGREEN_EX = ""
            RESET = ""


        class Style:
            RESET_ALL = ""
    else:
        print(
            f"{Fore.LIGHTWHITE_EX}Now showing {Fore.RED}c{Fore.MAGENTA}o{Fore.YELLOW}l{Fore.LIGHTGREEN_EX}o{Fore.BLUE}r{Fore.LIGHTBLUE_EX}e{Fore.LIGHTMAGENTA_EX}d {Fore.LIGHTWHITE_EX}text art!{Style.RESET_ALL}")

    print(rf"""    
                                                {Fore.RED}+{Fore.LIGHTRED_EX}--{Fore.RED}+
              {Fore.LIGHTRED_EX}__ _    {Fore.LIGHTBLUE_EX}___            {Fore.LIGHTBLUE_EX}_  __      {Fore.LIGHTRED_EX}|{Fore.LIGHTWHITE_EX}><{Fore.LIGHTRED_EX}|
    {Fore.LIGHTRED_EX} ___ ___ {Fore.LIGHTRED_EX}/{Fore.RED} _{Fore.LIGHTRED_EX}| {Fore.LIGHTRED_EX}|_ {Fore.LIGHTBLUE_EX}/ {Fore.BLUE}_ {Fore.LIGHTBLUE_EX}\ _  _ __ _| |/ {Fore.LIGHTBLUE_EX}/___  {Fore.RED} +{Fore.LIGHTRED_EX}--{Fore.RED}+
    {Fore.RED}(_-</ _ \  _|  {Fore.LIGHTRED_EX}_{Fore.BLUE}| (_) | || / _` | ' </ -{Fore.LIGHTBLUE_EX}_)  {Fore.LIGHTRED_EX}|{Fore.LIGHTWHITE_EX}><{Fore.LIGHTRED_EX}|
    {Fore.RED}/__/\___/_|  \__{Fore.LIGHTRED_EX}|{Fore.BLUE}\__\_\\_,_\__,_|_|\_\___{Fore.LIGHTBLUE_EX}|  {Fore.RED}+{Fore.LIGHTRED_EX}--{Fore.RED}+
    {Fore.RED}@@@@@@@@@@@@@@@@@{Fore.BLUE}@@@@@@@@@@@@@@@@@@@@@@@@@  {Fore.LIGHTRED_EX}|{Fore.LIGHTWHITE_EX}><{Fore.LIGHTRED_EX}|
                                             {Fore.LIGHTBLUE_EX}_.->-v-^._.
                                            {Fore.BLUE}/ "  .  ' . {Fore.LIGHTBLUE_EX}\.
     {Fore.RED}{Back.LIGHTYELLOW_EX}Softbody {Fore.BLUE}Earthquake{Back.BLUE} {Fore.LIGHTWHITE_EX}simulation in the command{Style.RESET_ALL}        
     {Back.BLUE}line with fixed presets, visualization video{Style.RESET_ALL}         
                {Back.BLUE}and useful figures.{Style.RESET_ALL}                                  
     {Style.RESET_ALL}
     """)

    structure = pyip.inputMenu(
        ["Box", "House", "Rhombus", "Hollow"],
        prompt=f"{Fore.LIGHTWHITE_EX}Select a {Fore.RED}softbody {Fore.LIGHTWHITE_EX}structure preset:\n",
        lettered=True,
    )
    print("      Rough Topology Diagram")

    if structure == "Box":
        print(
            rf"""
            {Fore.RED}O{Fore.LIGHTRED_EX}--.{Fore.RED}O{Fore.LIGHTRED_EX}.--{Fore.RED}O
            {Fore.LIGHTRED_EX}| {Fore.LIGHTWHITE_EX}/ {Fore.LIGHTRED_EX}| {Fore.LIGHTWHITE_EX}\ {Fore.LIGHTRED_EX}|    
            {Fore.RED}O{Fore.LIGHTRED_EX}:--{Fore.RED}O{Fore.LIGHTRED_EX}--:{Fore.RED}O
            {Fore.LIGHTRED_EX}| {Fore.LIGHTWHITE_EX}\ {Fore.LIGHTRED_EX}| {Fore.LIGHTWHITE_EX}/ {Fore.LIGHTRED_EX}|
            {Fore.RED}O{Fore.LIGHTRED_EX}--:{Fore.RED}O{Fore.LIGHTRED_EX}:--{Fore.RED}O
            {Fore.LIGHTRED_EX}| {Fore.LIGHTWHITE_EX}/ {Fore.LIGHTRED_EX}| {Fore.LIGHTWHITE_EX}\ {Fore.LIGHTRED_EX}|   
            {Fore.RED}O{Fore.LIGHTRED_EX}:--{Fore.RED}O{Fore.LIGHTRED_EX}--:{Fore.RED}O
            {Fore.LIGHTRED_EX}| {Fore.LIGHTWHITE_EX}\ {Fore.LIGHTRED_EX}| {Fore.LIGHTWHITE_EX}/ {Fore.LIGHTRED_EX}|    
            {Fore.RED}O{Fore.LIGHTRED_EX}--:{Fore.RED}O{Fore.LIGHTRED_EX}:--{Fore.RED}O   
            {Fore.LIGHTRED_EX}| {Fore.LIGHTWHITE_EX}/ {Fore.LIGHTRED_EX}| {Fore.LIGHTWHITE_EX}\ {Fore.LIGHTRED_EX}|   
            {Fore.RED}O{Fore.LIGHTRED_EX}'--{Fore.RED}O{Fore.LIGHTRED_EX}--'{Fore.RED}O{Style.RESET_ALL}   
          {Fore.LIGHTWHITE_EX}{Back.BLUE}O===========O{Style.RESET_ALL}
            """
        )
        points = np.array(
            [
                [-1, 0],
                [0, 0],
                [1, 0],
                [-1, 1],
                [0, 1],
                [1, 1],
                [-1, 2],
                [0, 2],
                [1, 2],
                [-1, 3],
                [0, 3],
                [1, 3],
                [-1, 4],
                [0, 4],
                [1, 4],
                [-1, 5],
                [0, 5],
                [1, 5],
                [-1, 6],
                [0, 6],
                [1, 6],
                [-1, 7],
                [0, 7],
                [1, 7],
            ]
        )
    elif structure == "House":
        print(
            rf"""
                  {Fore.RED}O
                 {Fore.LIGHTRED_EX}/ \
                {Fore.RED}O{Fore.LIGHTRED_EX}:-:{Fore.RED}O
               {Fore.LIGHTRED_EX}/ {Fore.LIGHTWHITE_EX}\ / {Fore.LIGHTRED_EX}\        
              {Fore.RED}O{Fore.LIGHTRED_EX}:-:{Fore.RED}O{Fore.LIGHTRED_EX}:-:{Fore.RED}O 
             {Fore.LIGHTRED_EX}/ {Fore.LIGHTWHITE_EX}\ / \ / {Fore.LIGHTRED_EX}\
            {Fore.RED}O{Fore.LIGHTRED_EX}:-'{Fore.RED}O{Fore.LIGHTRED_EX}'-:{Fore.RED}O{Fore.LIGHTRED_EX}:-'{Fore.RED}O       
            {Fore.LIGHTRED_EX}| {Fore.LIGHTWHITE_EX}\ {Fore.LIGHTRED_EX}| {Fore.LIGHTWHITE_EX}/ {Fore.LIGHTRED_EX}| {Fore.LIGHTWHITE_EX}\ {Fore.LIGHTRED_EX}|
            {Fore.RED}O{Fore.LIGHTRED_EX}--:{Fore.RED}O{Fore.LIGHTRED_EX}:--{Fore.RED}O{Fore.LIGHTRED_EX}--:{Fore.RED}O       
            {Fore.LIGHTRED_EX}| {Fore.LIGHTWHITE_EX}/ {Fore.LIGHTRED_EX}| {Fore.LIGHTWHITE_EX}\ {Fore.LIGHTRED_EX}| {Fore.LIGHTWHITE_EX}/ {Fore.LIGHTRED_EX}|
            {Fore.RED}O{Fore.LIGHTRED_EX}:--{Fore.RED}O{Fore.LIGHTRED_EX}--:{Fore.RED}O{Fore.LIGHTRED_EX}:--{Fore.RED}O        
            {Fore.LIGHTRED_EX}| {Fore.LIGHTWHITE_EX}\ {Fore.LIGHTRED_EX}| {Fore.LIGHTWHITE_EX}/ {Fore.LIGHTRED_EX}| {Fore.LIGHTWHITE_EX}\ {Fore.LIGHTRED_EX}|
            {Fore.RED}O{Fore.LIGHTRED_EX}--'{Fore.RED}O{Fore.LIGHTRED_EX}'--{Fore.RED}O{Fore.LIGHTRED_EX}--'{Fore.RED}O    
          {Fore.LIGHTWHITE_EX}{Back.BLUE}O===============O{Style.RESET_ALL}
            """
        )
        points = np.array(
            [
                [-2, 0],
                [-1, 0],
                [0, 0],
                [1, 0],
                [2, 0],
                [-2, 1],
                [-1, 1],
                [0, 1],
                [1, 1],
                [2, 1],
                [-2, 2],
                [-1, 2],
                [0, 2],
                [1, 2],
                [2, 2],
                [-2, 3],
                [-1, 3],
                [0, 3],
                [1, 3],
                [2, 3],
                [-1.5, 4],
                [-0.5, 4],
                [0.5, 4],
                [1.5, 4],
                [-1, 5],
                [0, 5],
                [1, 5],
                [-0.5, 6],
                [0.5, 6],
                [0, 7],
            ]
        )
    elif structure == "Rhombus":
        print(
            rf"""
                  {Fore.RED}O
                 {Fore.LIGHTRED_EX}/ \          
                {Fore.RED}O{Fore.LIGHTRED_EX}:-:{Fore.RED}O     
               {Fore.LIGHTRED_EX}/ {Fore.LIGHTWHITE_EX}\ / {Fore.LIGHTRED_EX}\ 
              {Fore.RED}O{Fore.LIGHTRED_EX}:-:{Fore.RED}O{Fore.LIGHTRED_EX}:-:{Fore.RED}O
             {Fore.LIGHTRED_EX}/ {Fore.LIGHTWHITE_EX}\ / \ / {Fore.LIGHTRED_EX}\
            {Fore.RED}O{Fore.LIGHTRED_EX}:-:{Fore.RED}O{Fore.LIGHTRED_EX}:-:{Fore.RED}O{Fore.LIGHTRED_EX}:-:{Fore.RED}O
             {Fore.LIGHTRED_EX}\ {Fore.LIGHTWHITE_EX}/ \ / \ {Fore.LIGHTRED_EX}/
              {Fore.RED}O{Fore.LIGHTRED_EX}:-:{Fore.RED}O{Fore.LIGHTRED_EX}:-:{Fore.RED}O
               {Fore.LIGHTRED_EX}\ {Fore.LIGHTWHITE_EX}/ \ {Fore.LIGHTRED_EX}/        
                {Fore.RED}O{Fore.LIGHTRED_EX}'-'{Fore.RED}O{Style.RESET_ALL}
             {Fore.LIGHTWHITE_EX}{Back.BLUE}O=========O{Style.RESET_ALL}
            """
        )
        points = np.array(
            [
                [-0.5, 0],
                [0.5, 0],
                [-1, 1],
                [0, 1],
                [1, 1],
                [-1.5, 2],
                [-0.5, 2],
                [0.5, 2],
                [1.5, 2],
                [-2, 3],
                [-1, 3],
                [0, 3],
                [1, 3],
                [2, 3],
                [-1.5, 4],
                [-0.5, 4],
                [0.5, 4],
                [1.5, 4],
                [-1, 5],
                [0, 5],
                [1, 5],
                [-0.5, 6],
                [0.5, 6],
                [0, 7],
            ]
        )
    elif structure == "Hollow":
        print(
          rf"""
          {Fore.RED}O{Fore.LIGHTRED_EX}--:{Fore.RED}O{Fore.LIGHTRED_EX}:--{Fore.RED}O{Fore.LIGHTRED_EX}--:{Fore.RED}O{Fore.LIGHTRED_EX}:--{Fore.RED}O
          {Fore.LIGHTRED_EX}| {Fore.LIGHTWHITE_EX}/ {Fore.LIGHTRED_EX}| {Fore.LIGHTWHITE_EX}\ {Fore.LIGHTRED_EX}| {Fore.LIGHTWHITE_EX}/ {Fore.LIGHTRED_EX}| {Fore.LIGHTWHITE_EX}\ {Fore.LIGHTRED_EX}|
          {Fore.RED}O{Fore.LIGHTRED_EX}:--{Fore.RED}O{Fore.LIGHTRED_EX}--:{Fore.RED}O{Fore.LIGHTRED_EX}:--{Fore.RED}O{Fore.LIGHTRED_EX}--:{Fore.RED}O       
          {Fore.LIGHTRED_EX}| {Fore.LIGHTWHITE_EX}\ {Fore.LIGHTRED_EX}|       {Fore.LIGHTRED_EX}| {Fore.LIGHTWHITE_EX}/ {Fore.LIGHTRED_EX}|
          {Fore.RED}O{Fore.LIGHTRED_EX}--:{Fore.RED}O       {Fore.RED}O{Fore.LIGHTRED_EX}:--{Fore.RED}O      
          {Fore.LIGHTRED_EX}| {Fore.LIGHTWHITE_EX}/ {Fore.LIGHTRED_EX}|       {Fore.LIGHTRED_EX}| {Fore.LIGHTWHITE_EX}\ {Fore.LIGHTRED_EX}|
          {Fore.RED}O{Fore.LIGHTRED_EX}:--{Fore.RED}O       {Fore.RED}O{Fore.LIGHTRED_EX}--:{Fore.RED}O    
          {Fore.LIGHTRED_EX}| {Fore.LIGHTWHITE_EX}\ {Fore.LIGHTRED_EX}|       {Fore.LIGHTRED_EX}| {Fore.LIGHTWHITE_EX}/ {Fore.LIGHTRED_EX}| 
          {Fore.RED}O{Fore.LIGHTRED_EX}--'{Fore.RED}O       {Fore.RED}O{Fore.LIGHTRED_EX}:--{Fore.RED}O{Style.RESET_ALL} 
        {Fore.LIGHTWHITE_EX}{Back.BLUE}O===================O{Style.RESET_ALL}
          """
        )
        points = np.array(
            [
                [-2, 0],
                [-1, 0],
                [1, 0],
                [2, 0],
                [-2, 1],
                [-1, 1],
                [1, 1],
                [2, 1],
                [-2, 2],
                [-1, 2],
                [1, 2],
                [2, 2],
                [-2, 3],
                [-1, 3],
                [1, 3],
                [2, 3],
                [-2, 4],
                [-1, 4],
                [0, 4],
                [1, 4],
                [2, 4],
                [-2, 5],
                [-1, 5],
                [0, 5],
                [1, 5],
                [2, 5],
            ]
        )
    else:
        points = np.array([])

    sleep(0.5)
    stiffness = pyip.inputMenu(
        ["Low", "Medium", "High"],
        prompt=f"{Fore.LIGHTWHITE_EX}Select the {Fore.RED}spring {Fore.LIGHTWHITE_EX}stiffness coefficient{Fore.LIGHTWHITE_EX}:\n",
        lettered=True
    )

    if stiffness == "Low":
        stiffness = 4e6
    elif stiffness == "Medium":
        stiffness = 8e6
    elif stiffness == "High":
        stiffness = 6e7
    else:
        stiffness = 0

    print("     Spring Stiffness Diagram")
    print(
        rf"""
      {Fore.LIGHTWHITE_EX}{Back.BLUE}D{Style.RESET_ALL}{Fore.RED}---^{Fore.LIGHTRED_EX}\/\/\/\/\/\/{Fore.RED}^---{Fore.LIGHTWHITE_EX}{Back.BLUE}O{Style.RESET_ALL} : {Fore.LIGHTGREEN_EX}{stiffness:.2e} {Fore.LIGHTGREEN_EX}N/m{Fore.LIGHTWHITE_EX}
        """
    )

    sleep(0.5)
    dampening = pyip.inputMenu(
        ["Low", "Medium", "High"],
        prompt=f"{Fore.LIGHTWHITE_EX}Select the {Fore.RED}damper {Fore.LIGHTWHITE_EX}dampening coefficient{Fore.LIGHTWHITE_EX}:\n",
        lettered=True
    )

    if dampening == "Low":
        dampening = 5e3
    elif dampening == "Medium":
        dampening = 1e4
    elif dampening == "High":
        dampening = 2e4
    else:
        dampening = 0

    print("     Damper Dampening Diagram")
    print(
        rf"""
      {Fore.LIGHTWHITE_EX}{Back.BLUE}D{Style.RESET_ALL}{Fore.RED}--------[{Fore.LIGHTRED_EX}::{Fore.RED}|--------{Fore.LIGHTWHITE_EX}{Back.BLUE}O{Style.RESET_ALL} : {Fore.LIGHTGREEN_EX}{dampening:.2e} {Fore.LIGHTGREEN_EX}N*s/m{Fore.LIGHTWHITE_EX}
        """
    )

    sleep(0.5)
    frequency = pyip.inputMenu(
        ["Low", "Medium", "High"],
        prompt=f"{Fore.LIGHTWHITE_EX}Select the plate horizontal vibration signal by frequency:\n",
        lettered=True,
    )
    amplitude = 0.0

    if frequency == "Low":
        frequency = 0.25
        amplitude = 2.0
    elif frequency == "Medium":
        frequency = 2.0
        amplitude = 0.4
    elif frequency == "High":
        frequency = 10.0
        amplitude = 0.05
    else:
        frequency = 0

    print("    Plate Vibration Diagram")
    print(
        rf"""
        {Fore.BLUE} .{Fore.LIGHTBLUE_EX}_________.  {Fore.LIGHTWHITE_EX}  : {Fore.LIGHTGREEN_EX}{frequency:.2f} {Fore.LIGHTGREEN_EX}Hz
      {Fore.LIGHTWHITE_EX}{Back.BLUE}<--{Style.RESET_ALL}{Fore.BLUE}|_________{Fore.LIGHTBLUE_EX}|{Fore.LIGHTWHITE_EX}{Back.BLUE}-->{Style.RESET_ALL} {Fore.LIGHTWHITE_EX}: {Fore.LIGHTGREEN_EX}{amplitude:.2f} {Fore.LIGHTGREEN_EX}m{Fore.LIGHTWHITE_EX}
        """
    )

    fps = 60
    ipf = 100
    delta = 1 / (fps * ipf)
    time = 0.0
    etime = 10
    shot = 0

    earth = 9.8

    plate = RigidPlate(sines=[], width=0, nodes=[])

    if structure == "Box":
        plate.width = 4
    elif structure == "House":
        plate.width = 5
    elif structure == "Rhombus":
        plate.width = 2
    elif structure == "Hollow":
        plate.width = 5.5

    sines = [Sine(frequency=frequency, amplitude=amplitude)]

    plate.sines = sines

    nodes = []

    for point in points:
        node = Node(mass=1e3, position=Vector(point[0], point[1]))
        nodes.append(node)

    if structure == "Box":
        plate.nodes.extend(nodes[0:3])
    elif structure == "House":
        plate.nodes.extend(nodes[0:5])
    elif structure == "Rhombus":
        plate.nodes.extend(nodes[0:2])
    elif structure == "Hollow":
        plate.nodes.extend(nodes[0:4])

    plate.set_kinematics(time)
    plate.set_nodes(0.8)

    sleep(0.5)
    loads = pyip.inputMenu(["No", "Yes"], prompt=f"{Fore.LIGHTWHITE_EX}Apply external loads?\n", lettered=True)

    if loads == "Yes":
        loads = []
        if structure == "Box":
            loads.extend(
                [
                    Load(node=nodes[3], force=Vector(20000, 0)),
                    Load(node=nodes[6], force=Vector(20000, 0)),
                    Load(node=nodes[9], force=Vector(20000, 0)),
                    Load(node=nodes[12], force=Vector(20000, 0)),
                    Load(node=nodes[15], force=Vector(20000, 0)),
                    Load(node=nodes[18], force=Vector(20000, 0)),
                    Load(node=nodes[21], force=Vector(20000, 0)),
                ]
            )
        elif structure == "House":
            loads.extend([Load(node=nodes[-1], force=Vector(120000, 0))])
        elif structure == "Rhombus":
            loads.extend(
                [
                    Load(node=nodes[-4], force=Vector(0, -60000)),
                    Load(node=nodes[-6], force=Vector(0, -60000)),
                ]
            )
        elif structure == "Hollow":
            loads.extend(
                [
                    Load(node=nodes[-2], force=Vector(0, -40000)),
                    Load(node=nodes[-4], force=Vector(0, -40000)),
                ]
            )
    else:
        loads = []

    if structure == "Box":
        sensor = Sensor(node=nodes[-2])
    elif structure == "House":
        sensor = Sensor(node=nodes[-4])
    elif structure == "Rhombus":
        sensor = Sensor(node=nodes[-1])
    elif structure == "Hollow":
        sensor = Sensor(node=nodes[-2])
    else:
        sensor = Sensor(node=nodes[0])

    try:
        if structure != "Hollow":
            delaunay = Delaunay(points)
            simplices = delaunay.simplices
        else:
            delaunay = None
            simplices = [[0, 1, 5], [0, 4, 5], [4, 5, 9], [4, 8, 9],
                         [8, 9, 13], [8, 12, 13], [12, 13, 17], [12, 16, 17],
                         [16, 17, 22], [16, 21, 22],
                         [2, 3, 7], [2, 6, 7], [6, 7, 11], [6, 10, 11],
                         [10, 11, 15], [10, 14, 15], [14, 15, 20], [14, 19, 20],
                         [19, 20, 25], [19, 24, 25], [17, 18, 23], [17, 22, 23],
                         [18, 19, 24], [18, 23, 24]]
    except Exception:
        print(f"{Fore.RED}Error computing the Delaunay triangulation.{Style.RESET_ALL}")
        raise Exception

    links: list[Link] = []
    triangles = []

    try:
        for simplex in simplices:

            def add_link_maybe(n1, n2):
                for linkm in links:
                    if (linkm.nodes[0] == nodes[n1] and linkm.nodes[1] == nodes[n2]) or (
                            linkm.nodes[0] == nodes[n2] and linkm.nodes[1] == nodes[n1]
                    ):
                        return linkm
                linkm = Link(
                    nodes=(nodes[n1], nodes[n2]), stiffness=stiffness, dampening=dampening
                )
                links.append(linkm)
                return linkm


            link1 = add_link_maybe(simplex[0], simplex[1])
            link2 = add_link_maybe(simplex[1], simplex[2])
            link3 = add_link_maybe(simplex[2], simplex[0])

            triangle = (
                [nodes[simplex[0]], nodes[simplex[1]], nodes[simplex[2]]],
                [link1, link2, link3],
            )
            triangles.append(triangle)
    except Exception:
        print(f"{Fore.RED}Error finding the simplices in the selected structure.{Style.RESET_ALL}")
        raise Exception

    energies = []
    epotenergies = []
    kinenergies = []
    potenergies = []

    sleep(0.5)
    try:
        if not path.exists("output"):
            print("Making the output folder.")
            mkdir("output")

        if not path.exists("output/frames"):
            print("Making the frames folder.")
            mkdir("output/frames")
        elif path.isdir("output/frames"):
            print("Removing the contents of the frames folder.")
            files = glob("output/frames/*")
            for file in files:
                remove(file)
    except Exception:
        print(f"{Fore.RED}Error creating output folder.{Style.RESET_ALL}")
        raise Exception
    sleep(0.5)
    print("Starting the simulation physics and animation loops.")
    sleep(0.5)
    print("Leaping through the time dimension with Verlet's method.")
    print(f"{Fore.LIGHTWHITE_EX}      markers: 1/4   1/2   3/4")
    print(f"{Fore.LIGHTWHITE_EX}                v     v     v")

    for t in range(etime):
        for s in range(fps):
            p = floor(20 * (time / etime))
            print(
                f"{Fore.LIGHTWHITE_EX}Progress {Fore.LIGHTWHITE_EX}: [{Back.RED}{Fore.LIGHTWHITE_EX}{'-' * p}*{'~' * (20 - p)}{Fore.LIGHTWHITE_EX}{Style.RESET_ALL}] : {Fore.RED}Wait{Fore.LIGHTWHITE_EX} ({round(100 * time / etime)}%){Fore.LIGHTWHITE_EX}",
                end="\r")
            for i in range(ipf):
                plate.set_kinematics(time)
                plate.set_nodes(0.8)

                for node in nodes:
                    node.force.set(Vector(0, -earth * node.mass))

                for load in loads:
                    if load.node not in plate.nodes:
                        load.node.force += load.force

                for link in links:
                    try:
                        force = link.get_force()
                    except ZeroDivisionError:
                        force = 0.0
                    try:
                        unit = link.get_unit()
                    except ZeroDivisionError:
                        unit = Vector(0, 0)
                    link.nodes[0].force -= unit * force
                    link.nodes[1].force += unit * force

                for n1, node1 in enumerate(nodes):
                    for n2 in range(n1 + 1, len(nodes)):
                        node2 = nodes[n2]
                        dist = Vector.dist(node1.position, node2.position)

                        if 0.01 < dist < 0.75:
                            try:
                                vector = Vector((node1.position.x - node2.position.x) / dist,
                                                (node1.position.y - node2.position.y) / dist)
                            except ZeroDivisionError:
                                vector = Vector(0, 0)
                            dx = 0.75 - dist
                            node1.force.x += 4e6 * dx * vector.x
                            node1.force.y += 4e6 * dx * vector.y
                            node2.force.x -= 4e6 * dx * vector.x
                            node2.force.y -= 4e6 * dx * vector.y

                for node in nodes:
                    if node not in plate.nodes:
                        acceleration = node.acceleration.copy()
                        node.acceleration = node.force / node.mass
                        node.position += (
                                node.velocity * delta + 0.5 * node.acceleration * delta ** 2
                        )
                        node.velocity += 0.5 * (acceleration + node.acceleration) * delta

                energy = 0.0
                epotenergy = 0.0
                kinenergy = 0.0
                potenergy = 0.0

                for link in links:
                    epotenergy += 0.5 * link.stiffness * link.get_displacement() ** 2

                for node in nodes:
                    kinenergy += 0.5 * node.mass * node.velocity.len() ** 2

                for node in nodes:
                    potenergy += 9.8 * node.mass * (node.position.y - 2)  # i know this is lying a bit

                energy += epotenergy + kinenergy + potenergy
                energies.append(energy)
                epotenergies.append(epotenergy)
                kinenergies.append(kinenergy)
                potenergies.append(potenergy)

                if sensor is not None:
                    sensor.record(time)

                time += delta

            try:
                surface = cairo.ImageSurface(cairo.FORMAT_RGB24, 1000, 1000)
                context = cairo.Context(surface)

                context.save()
                context.translate(0, 500)
                context.scale(1, -1)
                context.translate(0, -500)

                context.rectangle(0, 0, 1000, 1000)
                context.set_source_rgb(0, 1, 1)
                context.fill()

                context.rectangle(0, 0, 1000, 200)
                context.set_source_rgb(0, 1, 0)
                context.fill()

                context.scale(100, 100)
                context.translate(5, 5)
                context.translate(0, -4)

                context.rectangle(
                    plate.position.x - 0.5 * plate.width,
                    plate.position.y - 0.5,
                    plate.width,
                    0.5,
                )
                context.set_source_rgb(0.5, 0.25, 0.125)
                context.fill_preserve()
                context.set_line_width(0.1)
                context.set_source_rgb(0, 0, 0)
                context.stroke()

                for triangle in triangles:
                    link1, link2, link3 = triangle[1][0], triangle[1][1], triangle[1][2]
                    natural_semi = 0.5 * (link1.length + link2.length + link3.length)
                    actual_semi = 0.5 * (
                            link1.get_length() + link2.get_length() + link3.get_length()
                    )
                    natural_area = sqrt(
                        natural_semi
                        * (natural_semi - link1.length)
                        * (natural_semi - link2.length)
                        * (natural_semi - link3.length)
                    )
                    actual_area = sqrt(
                        actual_semi
                        * (actual_semi - link1.get_length())
                        * (actual_semi - link2.get_length())
                        * (actual_semi - link3.get_length())
                    )
                    try:
                        ratio = actual_area / natural_area
                    except ZeroDivisionError:
                        ratio = 1

                    context.move_to(triangle[0][0].position.x, triangle[0][0].position.y)
                    context.line_to(triangle[0][1].position.x, triangle[0][1].position.y)
                    context.line_to(triangle[0][2].position.x, triangle[0][2].position.y)
                    context.close_path()
                    if ratio < 1:
                        context.set_source_rgb(
                            1,
                            min(max(20 * (ratio - 1) + 1, 0), 1),
                            min(max(20 * (ratio - 1) + 1, 0), 1),
                        )
                    else:
                        context.set_source_rgb(
                            min(max(20 * (1 - ratio) + 1, 0), 1),
                            min(max(20 * (1 - ratio) + 1, 0), 1),
                            1,
                        )
                    context.fill()

                for link in links:
                    context.move_to(link.nodes[0].position.x, link.nodes[0].position.y)
                    context.line_to(link.nodes[1].position.x, link.nodes[1].position.y)
                    context.set_line_width(0.1)
                    context.set_source_rgb(0, 0, 0)
                    context.stroke()

                for node in nodes:
                    context.arc(node.position.x, node.position.y, 0.15, 0, tau)
                    context.set_source_rgb(1, 1, 1)
                    context.fill_preserve()
                    context.set_line_width(0.1)
                    context.set_source_rgb(0, 0, 0)
                    context.stroke()

                for load in loads:
                    if load.force.len() < 1e-5:
                        continue

                    context.save()
                    context.translate(load.node.position.x, load.node.position.y)
                    context.rotate(atan2(-load.force.y, -load.force.x))
                    context.translate(0.3, 0)
                    context.move_to(0, 0)
                    context.line_to(0.3, 0.3)
                    context.line_to(0.3, 0.08)
                    context.line_to(0.9, 0.12)
                    context.line_to(0.9, -0.12)
                    context.line_to(0.3, -0.08)
                    context.line_to(0.3, -0.3)
                    context.close_path()
                    context.set_source_rgb(1, 1, 0)
                    context.fill_preserve()
                    context.set_line_width(0.1)
                    context.set_source_rgb(0, 0, 0)
                    context.stroke()
                    context.restore()

                if sensor is not None:
                    context.save()
                    context.translate(sensor.node.position.x, sensor.node.position.y)
                    context.move_to(0.3, 0)
                    context.line_to(0, 0)
                    context.line_to(0, 0.3)
                    context.line_to(0, 0)
                    context.line_to(-0.3, 0)
                    context.line_to(0, 0)
                    context.line_to(0, -0.3)
                    context.line_to(0, 0)
                    context.line_to(0.3, 0)
                    context.arc(0, 0, 0.3, 0, tau)
                    context.set_line_width(0.2)
                    context.set_source_rgb(0, 0, 0)
                    context.stroke_preserve()
                    context.set_line_width(0.05)
                    context.set_source_rgb(1, 1, 0)
                    context.stroke()
                    context.restore()

                context.restore()

                if draw_text:
                    context.select_font_face("sans-serif", cairo.FontSlant.NORMAL, cairo.FontWeight.BOLD)
                    context.set_font_size(50)
                    context.move_to(20, 70)

                    for t, text in enumerate([f"shot={shot:05}", f"fps={fps}", f"ipf={ipf}", f"dt={delta:.2E}"]):
                        context.move_to(20, 70 + t * 60)
                        context.text_path(text)
                        context.set_source_rgb(1, 1, 0)
                        context.fill_preserve()
                        context.set_source_rgb(0, 0, 0)
                        context.set_line_width(2)
                        context.stroke()

                surface.write_to_png(f"output/frames/{shot:05}.png")
                surface.finish()
            except Exception:
                print(f"{Fore.RED}Error rendering with Cairo graphics library.{Style.RESET_ALL}")
                raise Exception
            shot += 1

    print(
        f"{Fore.LIGHTWHITE_EX}Progress {Fore.LIGHTWHITE_EX}: [{Back.LIGHTGREEN_EX}{Fore.LIGHTWHITE_EX}{'-' * 20}*{Style.RESET_ALL}] : {Fore.LIGHTGREEN_EX}Done{Fore.LIGHTWHITE_EX} (100%){Fore.LIGHTWHITE_EX}     ")
    sleep(0.5)
    print("\nAssembling the video file using the contents of the frames folder.")
    try:
        (
            ffmpeg.input("output/frames/%05d.png", framerate=fps)
            .output("output/video.mp4")
            .run(overwrite_output=True, quiet=True)
        )
    except Exception:
        print(f"{Fore.RED}Error encoding with FFmpeg media library.{Style.RESET_ALL}")
        raise Exception

    print("Removing the frames folder.")
    try:
        files = glob("output/frames/*")
        for file in files:
            remove(file)
        rmdir("output/frames")
    except Exception:
        print(f"{Fore.RED}Error deleting frames folder.{Style.RESET_ALL}")
        raise Exception

    sleep(0.5)
    print("Attempting signal processing on sensor data.")

    ts = np.array(sensor.times)
    ds = np.array(sensor.positions_x)
    vs = np.array(sensor.velocities_x)
    acs = np.array(sensor.accelerations_x)
    gs = acs / earth
    es = np.array(energies)
    epes = np.array(epotenergies)
    kes = np.array(kinenergies)
    pes = np.array(potenergies)

    try:
        plt.style.use("dark_background")
        fig1, (ax1, ax2, ax3) = plt.subplots(nrows=3)
        fig1.suptitle('Horizontal kinematics of the "sensor" node in time')
        ax1.plot(ts, ds, color="red")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Displacement (m)")
        ax2.plot(ts, vs, color="dodgerblue")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Velocity (m/s)")
        ax3.plot(ts, acs, color="magenta")
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Acceleration (m/s/s)")

        fig2, ax4 = plt.subplots()
        spectrogram = ax4.specgram(
            np.abs(acs[::100]), NFFT=32, Fs=1 / (100 * delta), noverlap=20, cmap="inferno"
        )[3]
        colorbar = fig2.colorbar(spectrogram, ax=ax4)
        ax4.set_title(
            'Spectrogram of power spectral density of\nhorizontal acceleration magnitude of the "sensor" node'
        )
        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel("Frequency (Hz)")
        colorbar.set_label("PSD of Acceleration")

        fig3, ax5 = plt.subplots()
        ax5.set_title("Energy of the structure in time.")
        ax5.plot(ts, es, color="yellow")
        ax5.plot(ts, epes, color="pink")
        ax5.plot(ts, kes, color="red")
        ax5.plot(ts, pes, color="blue")
        ax5.set_xlabel("Time (s)")
        ax5.set_ylabel("Energy (J)")
        ax5.legend(["Combined Energy", "Elastic Potential Energy", "Kinetic Energy", "Potential Energy"])
        fig3.savefig("output/figure3.png")

        sleep(0.5)
        print("Making the figures.")
        fig1.savefig("output/figure1.png")
        fig2.savefig("output/figure2.png")
        fig3.savefig("output/figure3.png")
        print("Output can now be found in the output folder of the current working directory")
    except Exception:
        print(f"{Fore.RED}Error saving graphs with matplotlib plotting library.{Style.RESET_ALL}")
        raise Exception
except KeyboardInterrupt:
    exit()
