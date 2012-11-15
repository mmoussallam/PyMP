#*****************************************************************************/
#                                                                            */
#                           pymp_Log.py                                      */
#                                                                            */
#                        Matching Pursuit Library                            */
#                                                                            */
# M. Moussallam                                             Mon Aug 16 2010  */
# -------------------------------------------------------------------------- */
#                                                                            */
#                                                                            */
#  This program is free software; you can redistribute it and/or             */
#  modify it under the terms of the GNU General Public License               */
#  as published by the Free Software Foundation; either version 2            */
#  of the License, or (at your option) any later version.                    */
#                                                                            */
#  This program is distributed in the hope that it will be useful,           */
#  but WITHOUT ANY WARRANTY; without even the implied warranty of            */
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the             */
#  GNU General Public License for more details.                              */
#                                                                            */
#  You should have received a copy of the GNU General Public License         */
#  along with this program; if not, write to the Free Software               */
#  Foundation, Inc., 59 Temple Place - Suite 330,                            */
#  Boston, MA  02111-1307, USA.                                              */
#                                                                            */
#******************************************************************************/

'''
Module pymp_Log
===============
                                                                          
'''

import logging
import sys , traceback
import matplotlib.pyplot as plt

Levels = {-1: logging.NOTSET,
          0:logging.ERROR,
          1:logging.WARNING,
          2:logging.DEBUG,
          3:logging.INFO};



class pymp_Log():
    """ This object should be made static and used to Log warnings and error messages 
    debug Levels are handled that way:
        - level -1 : no messages at all 
        - level 0 : only error messages are shown
        - level 1 : error and warning messages
        - level 2 : error , warnings and debug messages
        - level 3 : error , warnings , debug and info messages
    the imode attribute sets matplotlib drawing mode to interactive"""
        
    # attributes
    name = None;
    outputFile = None;
    debugLevel = 0;
    imode = True;
    logger = None
    noCtx = False;
    
    # constructor
    def __init__(self ,loggerName,  level=0 , outputFilePath = None , imode = False , noContext=False):
        """ constructor """
        self.name = loggerName;
        self.outputFile = outputFilePath;
        self.debugLevel = level;
        self.imode = imode;
        self.noCtx = noContext;
        # create logger
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(Levels[self.debugLevel])
        
        # create console handler and set level to debug
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        
        # create formatter
        
        formatter = logging.Formatter(" %(levelname)s - %(message)s")

        # add formatter to ch
        ch.setFormatter(formatter)
        
        # add ch to logger
        self.logger.addHandler(ch)
        
        # "application" code        
        if self.outputFile is None:
            self.logger.info("created log handler with no file output for function : " + self.name)
        else:
            self.logger.info("created log handler for function : " + self.name)
        
        if self.imode:
            plt.ion()
            self.logger.info("Set interactive mode ON for pyplot" )
    
    def setLevel(self , newLevel):
        self.debugLevel = newLevel;
        self.logger.setLevel(Levels[self.debugLevel])
    
    
    def getContext(self):
        if self.noCtx:
            return '';
        list = traceback.extract_stack(limit=3)
        str_context = list[0][0].split('/')[-1]+ '::'+str(list[0][1]) + '::'+ list[0][2] +'::';
        return str_context
        
            
    def info(self , message):
        self.logger.info(self.getContext() + message)
    
    def debug(self , message):
        self.logger.debug(self.getContext() + message)
        
    def warning(self , message):
        self.logger.warning(self.getContext() + message)
        
    def error(self , message):
        self.logger.error(self.getContext() + message)
        