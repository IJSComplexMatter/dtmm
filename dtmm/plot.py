"""plotting functions and classes"""
from __future__ import division, print_function, absolute_import

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from dtmm.ray import field2ray, field2ray2
from dtmm.tmm import ipropagate, ipropagate2
from dtmm.color import load_tcmf, specter2color

class field_viewer(object):       
    def __init__(self,xfield, yfield, refind = 1.5, beta = 0., phi = 0., k0 = [1], nstep = 16, maxfocus = 100):
        self.xfield = xfield
        self.yfield = yfield
        self.analizer = (0,1)
        self.polarizer = (1,0)
        self.refind = refind
        self.beta = beta
        self.phi = phi
        self.k0 = k0
        self.nstep = nstep
        self.intensity = 1.
        self.cmf = load_tcmf()
        self.n = len(xfield)
        
        self.fig, self.ax1 = plt.subplots()
        
        plt.subplots_adjust(bottom=0.25)
        
        self.ixfield = xfield
        self.iyfield = yfield
        self.xray = field2ray(xfield, pol = self.analizer, phi = self.phi)
        self.yray = field2ray(yfield, pol = self.analizer, phi = self.phi)
        self.ray = [self.xray[i]*self.polarizer[0]+self.yray[i]*self.polarizer[1] for i in range(self.n)]
        
        im = specter2color(self.ray,self.cmf, norm = self.intensity)
        self.imax = self.ax1.imshow(im)#, animated=True)
        
    
        self.axfocus = plt.axes([0.25, 0.14, 0.65, 0.03])
        self.axanalizer = plt.axes([0.25, 0.10, 0.65, 0.03])
        self.axpolarizer = plt.axes([0.25, 0.06, 0.65, 0.03])
        self.axintensity = plt.axes([0.25, 0.02, 0.65, 0.03])
        self.sfocus = Slider(self.axfocus, "focus",-maxfocus,maxfocus,valinit = 0., valfmt='%.1f')
        self.sintensity = Slider(self.axintensity, "light",0,10,valinit = 1., valfmt='%.1f')
        self.sanalizer = Slider(self.axanalizer, "analizer",-180,180,valinit = 0, valfmt='%.1f')
        self.spolarizer = Slider(self.axpolarizer, "polarizer",-180,180,valinit = 0, valfmt='%.1f')
                
        def update_focus(d):
            self.ixfield = ipropagate(self.xfield, d = d, refind = self.refind, beta = self.beta, phi = self.phi, k0 = self.k0, nstep = self.nstep)
            self.iyfield = ipropagate(self.yfield, d = d, refind = self.refind, beta = self.beta, phi = self.phi, k0 = self.k0, nstep = self.nstep)
            field = [self.ixfield[i]*self.polarizer[0]+self.iyfield[i]*self.polarizer[1] for i in range(self.n)]
            self.ray = field2ray(field, pol = self.analizer, phi = self.phi)
      
            im = specter2color(self.ray,self.cmf, norm = 1./self.intensity)
            self.imax.set_data(im)
            self.fig.canvas.draw_idle()
        
        def update_light(d):
            self.intensity = d
            im = specter2color(self.ray,self.cmf, norm = 1./self.intensity)
            self.imax.set_data(im)
            self.fig.canvas.draw_idle()  
            
        def update_analizer(d):
            self.analizer = np.cos(d/180*np.pi), np.sin(d/180*np.pi)
            field = [self.ixfield[i]*self.polarizer[0]+self.iyfield[i]*self.polarizer[1] for i in range(self.n)]
            self.ray = field2ray(field, pol = self.analizer, phi = self.phi)       
            
            im = specter2color(self.ray,self.cmf, norm = 1./self.intensity)
            self.imax.set_data(im)
            self.fig.canvas.draw_idle()  

        def update_polarizer(d):
            self.polarizer = np.cos(d/180*np.pi), np.sin(d/180*np.pi)
            field = [self.ixfield[i]*self.polarizer[0]+self.iyfield[i]*self.polarizer[1] for i in range(self.n)]
            self.ray = field2ray(field, pol = self.analizer, phi = self.phi)       
                     
            im = specter2color(self.ray,self.cmf, norm = 1./self.intensity)
            self.imax.set_data(im)
            self.fig.canvas.draw_idle() 
            

            
        self.ids1 = self.sfocus.on_changed(update_focus)
        self.ids2 = self.sintensity.on_changed(update_light)
        self.ids3 = self.sanalizer.on_changed(update_analizer)
        self.ids4 = self.spolarizer.on_changed(update_polarizer)
        #self.ids5 = self.srotation.on_changed(update_rotation)
        
    def show(self):
        plt.show()

#I am using a class instead of function, so that plot objects have references and not garbage collected in interactive mode        
class field_viewer2(object):       
    def __init__(self,field, refind = 1.5, beta = 0., phi = 0., k0 = [1], nstep = 16, maxfocus = 100):
        self.field = field
        self.pol = (1,0)
        self.refind = refind
        self.beta = beta
        self.phi = phi
        self.k0 = k0
        self.nstep = nstep
        self.intensity = 1.
        self.cmf = load_tcmf()
        
        self.fig, self.ax1 = plt.subplots()
        
        plt.subplots_adjust(bottom=0.25)
        
        self.ifield = field
        self.ray = field2ray2(field, pol = self.pol, phi = self.phi)
        im = specter2color(self.ray,self.cmf, norm = self.intensity)
        self.imax = self.ax1.imshow(im)#, animated=True)
        
    
        self.axfocus = plt.axes([0.25, 0.15, 0.65, 0.03])
        self.axanalizer = plt.axes([0.25, 0.10, 0.65, 0.03])
        self.axintensity = plt.axes([0.25, 0.05, 0.65, 0.03])
        self.sfocus = Slider(self.axfocus, "focus",-maxfocus,maxfocus,valinit = 0., valfmt='%.1f')
        self.sintensity = Slider(self.axintensity, "light",0,10,valinit = 1., valfmt='%.1f')
        self.sanalizer = Slider(self.axanalizer, "analizer",-180,180,valinit = 0, valfmt='%.1f')
        
        def update_focus(d):
            self.ifield = ipropagate2(self.field, d = d, refind = self.refind, 
                                      beta = self.beta, phi = self.phi, 
                                      k0 = self.k0, nstep = self.nstep,
                                      select = "forward")
            self.ray = field2ray2(self.ifield, pol = self.pol, phi = self.phi)
            im = specter2color(self.ray,self.cmf, norm = 1./self.intensity)
            self.imax.set_data(im)
            self.fig.canvas.draw_idle()
        
        def update_light(d):
            self.intensity = d
            im = specter2color(self.ray,self.cmf, norm = 1./self.intensity)
            self.imax.set_data(im)
            self.fig.canvas.draw_idle()  
            
        def update_analizer(d):
            print (d)
            self.pol = np.cos(d/180*np.pi), np.sin(d/180*np.pi)
            self.ray = field2ray2(self.ifield, pol = self.pol, phi = self.phi)
            im = specter2color(self.ray,self.cmf, norm = 1./self.intensity)
            self.imax.set_data(im)
            self.fig.canvas.draw_idle()  
            
        self.ids1 = self.sfocus.on_changed(update_focus)
        self.ids2 = self.sintensity.on_changed(update_light)
        self.ids3 = self.sanalizer.on_changed(update_analizer)
        
    def show(self):
        plt.show()
        
#         self.playax = plt.axes([0.8, 0.025, 0.1, 0.04])
#         self.playbutton = Button(self.playax, 'Play', hovercolor='0.975')
#         self.stopax = plt.axes([0.6, 0.025, 0.1, 0.04])
#         self.stopbutton = Button(self.stopax, 'Stop', hovercolor='0.975')
# 
#         def update_anim(i):
#             self.imax.set_data(norm(video[i]))
#             self.sindex.set_val(i)
#             return self.imax,
#          
#         def play_video(event):
#              #disconnect slider move event
#             self.sindex.disconnect(self.ids)
#             if hasattr(self, "ani"):
#                 self.ani.event_source.stop()
#             start = int(self.sindex.val)
#             if start == video.shape[0] -1:
#                 start = 0
#             self.ani = FuncAnimation(self.fig, update_anim, frames=range(start,video.shape[0]), blit=True, interval = 1, repeat = False)
#             
#         def stop_video(event):
#             self.ani.event_source.stop()
#             del self.ani
#             gc.collect()
#             self.ids = self.sindex.on_changed(update)
#     
#         self.playbutton.on_clicked(play_video)
#         self.stopbutton.on_clicked(stop_video)    
        #plt.show()

#def test_show_video():    
#    video = np.random.randint(0,255,(1024,256,256),"uint8")
#    plot = show_video(video)
#    
#class show_histogram(object):
#    """Shows video in plot. You need to hold reference to this object, otherwise it will not work in interactive mode.
#    """        
#    def __init__(self,video):
#        self.fig, self.ax1 = plt.subplots()
#        plt.subplots_adjust(bottom=0.25)
#        frame = video[0]
#        hist, n = np.histogram(frame.ravel(),255,(0,255))
#        self.hisl, = self.ax1.plot(hist)
#    
#        self.axindex = plt.axes([0.25, 0.15, 0.65, 0.03])
#        self.sindex = Slider(self.axindex, "frame",0,video.shape[0]-1,valinit = 0, valfmt='%i')
#        
#        def update(val):
#            i = int(self.sindex.val)
#            frame= video[i]
#            hist, n = np.histogram(frame.ravel(),255,(0,255))
#            self.hisl.set_ydata(hist)
#            self.fig.canvas.draw_idle()
#        
#        self.ids = self.sindex.on_changed(update)
#        #plt.show()    
#    
#def test_show_histogram():    
#    video = np.random.randint(0,255,(1024,256,256),"uint8")
#    plot = show_histogram(video)
#                
#class show_fftdata(object):
#    """Shows correlation data in plot. You need to hold reference to this object, otherwise it will not work in interactive mode.
#    """        
#    def __init__(self,data, semilogx = True, normalize = True, sector = 5, angle = 0, kstep = 1):
#        
#        self.shape = data.shape
#        self.data = data
#                
#        self.fig, (self.ax1,self.ax2) = plt.subplots(1,2,gridspec_kw = {'width_ratios':[3, 1]})
#        plt.subplots_adjust(bottom=0.25)
#        
#        graph_shape = self.shape[0], self.shape[1]
#        
#        graph = np.zeros(graph_shape)
#        
#        max_k_value = np.sqrt((graph_shape[0]//2+1)**2+ graph_shape[1]**2) *np.sqrt(2)
#        graph[0,0] = max_k_value 
#        
#        self.im = self.ax2.imshow(graph, extent=[0,self.shape[1],self.shape[0]//2+1,-self.shape[0]//2-1])
#        #self.ax2.grid()
#        
#        norm = data[:,:,0]
#        if normalize == False:
#            norm = np.ones_like(norm)
#        
#        if semilogx == True:
#            self.l, = self.ax1.semilogx(data[0,0,:]/norm[0,0])
#        else:
#            self.l, = self.ax1.plot(data[0,0,:]/norm[0,0])
#
#        self.kax = plt.axes([0.1, 0.15, 0.65, 0.03])
#        self.kindex = Slider(self.kax, "k",0,max_k_value,valinit = 0, valfmt='%i')
#
#        self.phiax = plt.axes([0.1, 0.10, 0.65, 0.03])
#        self.phiindex = Slider(self.phiax, "$\phi$",-90,90,valinit = angle, valfmt='%.2f')        
#
#        self.sectorax = plt.axes([0.1, 0.05, 0.65, 0.03])
#        self.sectorindex = Slider(self.sectorax, "sector",0,180,valinit = sector, valfmt='%.2f') 
#                                  
#        def update(val):
#            k = self.kindex.val
#            phi = self.phiindex.val
#            sector = self.sectorindex.val
#            ki = int(round(- k * np.sin(np.pi/180 * phi)))
#            kj =  int(round(k * np.cos(np.pi/180 * phi)))
#            #self.l.set_ydata(data[ki,kj,:]/norm[ki,kj])
#            # recompute the ax.dataLim
#            self.ax1.relim()
#
#            # update ax.viewLim using the new dataLim
#            self.ax1.autoscale_view()
#            graph = np.zeros((self.shape[0], self.shape[1]))
#            if sector != 0:
#                indexmap = sector_indexmap(self.shape[0],self.shape[1],phi, sector, kstep)
#            else:
#                indexmap = line_indexmap(self.shape[0],self.shape[1],phi)
#            #graph = np.concatenate((indexmap[:,:,np.newaxis],indexmap[:,:,np.newaxis],indexmap[:,:,np.newaxis]),-1) #make color image 
#            graph = indexmap
#            mask = (indexmap == int(round(k)))
#            graph[mask] = max_k_value +1
#            
#            
#            avg_data = correlate(self.data[mask,:]).mean(0)
#            self.l.set_ydata(avg_data/avg_data[0])
#            avg_graph = np.zeros(graph_shape)
#            avg_graph[mask] = 1
#            #graph[ki,kj] = 1
#            #print (ki,kj)
#            
#            self.im.set_data(np.fft.fftshift(graph,0))
#            
#            #circ = Circle((ki,kj),5)
#            #self.ax2.add_patch(circ)
#            self.fig.canvas.draw_idle()  
#        self.update = update      
#                        
#        self.kindex.on_changed(update)
#        self.phiindex.on_changed(update)
#        self.sectorindex.on_changed(update)
#        plt.show()
#
#class show_correlation(object):
#    """Shows correlation data in plot. You need to hold reference to this object, otherwise it will not work in interactive mode.
#    """        
#    def __init__(self,data, semilogx = True, normalize = True):
#        
#        self.shape = data.shape
#        self.data = data
#                
#        self.fig, (self.ax1,self.ax2) = plt.subplots(1,2,gridspec_kw = {'width_ratios':[3, 1]})
#        plt.subplots_adjust(bottom=0.25)
#        
#        graph_shape = self.shape[0], self.shape[1]
#        
#        graph = np.zeros(graph_shape)
#        
#        max_graph_value = max(graph_shape[0]//2+1, graph_shape[1]) 
#        graph[0,0] = max_graph_value 
#        
#        self.im = self.ax2.imshow(graph, extent=[0,self.shape[1],self.shape[0]//2+1,-self.shape[0]//2-1])
#        #self.ax2.grid()
#        
#        norm = data[:,:,0]
#        if normalize == False:
#            norm = np.ones_like(norm)
#        
#        if semilogx == True:
#            self.l, = self.ax1.semilogx(data[0,0,:]/norm[0,0])
#        else:
#            self.l, = self.ax1.plot(data[0,0,:]/norm[0,0])
#
#        self.kax = plt.axes([0.1, 0.15, 0.65, 0.03])
#        self.kindex = Slider(self.kax, "k",0,data.shape[1]-1,valinit = 0, valfmt='%i')
#
#        self.phiax = plt.axes([0.1, 0.10, 0.65, 0.03])
#        self.phiindex = Slider(self.phiax, "$\phi$",-90,90,valinit = 0, valfmt='%.2f')        
#
#        self.sectorax = plt.axes([0.1, 0.05, 0.65, 0.03])
#        self.sectorindex = Slider(self.sectorax, "sector",0,180,valinit = 5, valfmt='%.2f') 
#                                  
#        def update(val):
#            k = self.kindex.val
#            phi = self.phiindex.val
#            sector = self.sectorindex.val
#            ki = int(round(- k * np.sin(np.pi/180 * phi)))
#            kj =  int(round(k * np.cos(np.pi/180 * phi)))
#            self.l.set_ydata(data[ki,kj,:]/norm[ki,kj])
#            # recompute the ax.dataLim
#            self.ax1.relim()
#
#            # update ax.viewLim using the new dataLim
#            self.ax1.autoscale_view()
#            graph = np.zeros((self.shape[0], self.shape[1]))
#            if sector != 0:
#                indexmap = sector_indexmap(self.shape[0],self.shape[1],phi, sector, 1)
#            else:
#                indexmap = line_indexmap(self.shape[0],self.shape[1],phi)
#            #graph = np.concatenate((indexmap[:,:,np.newaxis],indexmap[:,:,np.newaxis],indexmap[:,:,np.newaxis]),-1) #make color image 
#            graph = indexmap
#            mask = (indexmap == int(round(k)))
#            graph[mask] = max_graph_value +1
#            
#            
#            avg_data = self.data[mask,:].mean(0)
#            self.l.set_ydata(avg_data/avg_data[0])
#            avg_graph = np.zeros(graph_shape)
#            avg_graph[mask] = 1
#            #graph[ki,kj] = 1
#            #print (ki,kj)
#            
#            self.im.set_data(np.fft.fftshift(graph,0))
#            
#            #circ = Circle((ki,kj),5)
#            #self.ax2.add_patch(circ)
#            self.fig.canvas.draw_idle()        
#                        
#        self.kindex.on_changed(update)
#        self.phiindex.on_changed(update)
#        self.sectorindex.on_changed(update)
#        plt.show()
#
#
#def test_show_correlation():    
#    video = np.random.randint(0,255,(256,129,1024*8),"uint8")
#    return show_correlation(video)
#    
#    
#    
