import wx
import os
import time
class Mywin(wx.Frame): 
   def __init__(self, parent, title): 
      super(Mywin, self).__init__(parent, title = title,size = (600,400))
      panel = wx.Panel(self) 
      vbox = wx.BoxSizer() 
      self.SetBackgroundColour((51, 153, 255, 150))
         
      self.btn = wx.Button(panel,-1," Forecast ", pos =(250, 20), size=(100, 50))
      self.btn.SetBackgroundColour((150, 255, 153,200))
      # self.btn.SetSize((200, 100))
      # vbox.Add(self.btn) 
      self.btn.Bind(wx.EVT_BUTTON,self.OnClicked) 

      wx.SizerFlags.DisableConsistencyChecks() 
      self.tbtn = wx.ToggleButton(panel , -1, "Activate Auto-Forecast", size=(200, 50), pos=(200, 100))
      self.tbtn.SetBackgroundColour((255, 0, 102, 200))
      # vbox.Add(self.tbtn)
      self.tbtn.Bind(wx.EVT_TOGGLEBUTTON,self.OnToggle)

      self.btn = wx.Button(panel, -1, " Prediction ",size=(100, 50), pos=(250, 200))
      self.btn.SetBackgroundColour((255, 153, 0, 200))
      # vbox.Add(self.btn)
      self.btn.Bind(wx.EVT_BUTTON, self.OnClicked_2)

      hbox = wx.BoxSizer(wx.HORIZONTAL)
         

         
      vbox.Add(hbox,1,wx.ALIGN_CENTER) 
      panel.SetSizer(vbox) 
        
      self.Centre() 
      self.Show() 
      self.Fit()  
		
   def OnClicked(self, event): 
      btn = event.GetEventObject().GetLabel()
      os.system('py -3.6 -m Main_Forecast.py')
      print ("Finished Forecast by Button = ",btn)

   def OnClicked_2(self, event):
      btn = event.GetEventObject().GetLabel()
      os.system('py -3.6 -m Main_Prediction.py')
      print ("Finished Prediction by Button = ",btn)

   def OnToggle(self,event): 
      state = event.GetEventObject().GetValue() 
		
      if state == True:
         event.GetEventObject().SetLabel("Deactivate Auto-Prediction")
         while(state==True):
            print ("Enabled Auto-Prediction (daily update)")
            os.system('py -3.6 -m Main_Forecast.py')
            print("Finished daily update. Wait for next day.")
            i = 1
            while(i<20):
               print("Please wait "+str(20-i)+" seconds")
               time.sleep(1)
               i+=1
      else:
         print (" Disabled Auto-Prediction")
         event.GetEventObject().SetLabel("Activate Auto-Prediction")
             
app = wx.App() 
Mywin(None,  'Weather Prediction using Machine Learning') 
app.MainLoop()