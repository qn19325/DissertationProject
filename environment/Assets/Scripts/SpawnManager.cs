using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SpawnManager : MonoBehaviour
{
    TunnelSpawner tunnelSpawner;
    // Start is called before the first frame update
    void Start()
    {
        tunnelSpawner = GetComponent<TunnelSpawner>();
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    public void SpawnTriggeredEntered() 
    {
        tunnelSpawner.SpawnTunnel();
    }
}
