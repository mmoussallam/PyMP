#*****************************************************************************/
#                                                                            */
#                           pymp_Atom.py                                     */
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
Module pymp_Atom
================
                                                                          
'''

class pymp_Atom:
    """ Abstract Atom class interface: 
    
    To implement a new type of atom, you must derive from this class. 
    
    Attributes:
    
        - `nature`: A string describing the atom type (e.g MDCT, MCLT , GaborReal) default is MDCT
        
        - `length: Sample length of the atom (default is 0)
        
        - `timePosition`: The index of the first atom sample in a signal
        
        - `waveform`: a numpy array that will contain the atom waveform 
        
        - `amplitude`: the atom amplitude
        
        - `samplingFrequency`: the atom sampling frequency
        
    """ 

    # default values
    nature = 'Abstract';
    length = 0 ;
    timePosition = 0;
    waveform = None;
    samplingFrequency = 0;
    amplitude = 0;
    phase = None;
    def __init__(self  ):
        self.length = 0;
        self.amplitude = 0;
        self.timePosition = 0;
    
    # mandatory functions
    def getWaveform(self):
        # A function to retrieve the atom waveform 
        return self.waveform;
    


