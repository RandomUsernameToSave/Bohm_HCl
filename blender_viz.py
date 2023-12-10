import bpy
import numpy as np
import pickle

file_traj = open(r"C:\Users\cleme\Desktop\BohmSimulation\particle_trajectories.pkl",'rb')
Samples = pickle.load(file)
traj = np.asarray(pickle.load(file_traj))

def particleSetter(scene, degp):
    particle_systems = object.evaluated_get(degp).particle_systems
    particles = particle_systems[0].particles
    totalParticles = len(particles)

    scene = bpy.context.scene
    cFrame = scene.frame_current
    sFrame = scene.frame_start

    # at start-frame, clear the particle cache
    if cFrame == sFrame:
        psSeed = object.particle_systems[0].seed
        object.particle_systems[0].seed = psSeed

    # Rotate particles based on index (t_p) and frame (t_f)
    t_p = np.linspace(0, 2*np.pi, totalParticles, endpoint=False)
    t_f = cFrame / 20.0
    
    # data = np.asarray(Samples)
    data = np.asarray(traj[cFrame])
    flatList = data.ravel()

    # Set the location of all particle locations to flatList
    particles.foreach_set("location", flatList)
    

# Prepare particle system
object = bpy.data.objects["Cube"]
object.modifiers.new("ParticleSystem", 'PARTICLE_SYSTEM')
object.particle_systems[0].settings.count = 20000
object.particle_systems[0].settings.frame_start = 1
object.particle_systems[0].settings.frame_end = 1
object.particle_systems[0].settings.lifetime = 1000
object.show_instancer_for_viewport = False
degp = bpy.context.evaluated_depsgraph_get()

#clear the post frame handler
bpy.app.handlers.frame_change_post.clear()

#run the function on each frame
bpy.app.handlers.frame_change_post.append(particleSetter)

# Update to a frame where particles are updated
bpy.context.scene.frame_current = 2