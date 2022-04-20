using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

public class TunnelSpawner : MonoBehaviour
{
    public List<GameObject> tunnels;
    private float offset = 200f;
    // Start is called before the first frame update
    void Start()
    {
        if(tunnels != null && tunnels.Count > 0) 
        {
            // order list of tunnels by z coordinate
            tunnels = tunnels.OrderBy( t => t.transform.position.z).ToList();
        }
    }

    public void SpawnTunnel()
    {
        // SpawnTunnel works by moving the first tunnel in this list to the last tunnel in the list offsetting its z coordinate
        GameObject tunnelToMove = tunnels[0];
        tunnels.Remove(tunnelToMove);
        float newZCoord = tunnels[tunnels.Count - 1].transform.position.z + offset;
        tunnelToMove.transform.position = new Vector3(0, 0, newZCoord);
        tunnels.Add(tunnelToMove);
    }
}
