'''*****************************************************************************/
#                                                                            */
#                           pymp_Dico.py                               */
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
#*****************************************************************************'''


class pymp_Dico():
    """ This class creates an interface that any type of dictionary should reproduce 
        in order to be used correclty by Pursuit algorithm in this framework:
            - `sizes`: a list of scales 
            - `blocks`: a list of blocks that handles the projection of a residual signal along with
                        the selection of a projection maximum given a transform and a criteria
        """
    # attributes:    
    sizes = None;
    tolerances = None;
    blocks = None;
    overlap = 0.5;
    nature = 'Abstract'
    
    def __init__(self):
        """ default constructor doesn't do anything"""
        
    def getN(self):
        return self.sizes[-1] # last element of the list should be the biggest
    
