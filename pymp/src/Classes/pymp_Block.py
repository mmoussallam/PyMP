#*****************************************************************************/
#                                                                            */
#                           pymp_Block.py                              */
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
#*****************************************************************************
'''
pymp_Block
==========

A block is an object encapsulating the atomic projection operations

PyMP considers dictionaries as collections of block that are usually defined by 
a monoscale time-frequency transform (e.g. Gabor or MDCT)

Here we just define an abstract Block class, see mdct.pymp_MDCTBlock for a practical implementation

'''



class pymp_Block:
    ''' A block is an instance handling projections for Matching Pursuit. Mandatory fields:
        - type : the type of dictionary (e.g  Gabor , MDCT , Haar ...)  
        - scale : the scale of the block
        - residualSignal : a py_pursuit_Signal instance that describes the current residual
        Mandatory methods
        - update :  updates the inner products table
        - getMaximum : retrieve the maximum absolute value of inner products
        - getMaxAtom : return a corresponding Atom instance'''
    
    #members 
    scale = 0;    
    residualSignal = None;
    
    # methods
    def __init__(self):
        """ empty constructor """
    
    