using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PlayerController : MonoBehaviour
{
    private Vector3 startingPosition = new Vector3(0, 0.51f, 0);
    Rigidbody rBody;
    public float speed = 0;
    public SpawnManager spawnManager;
    

    // Start is called before the first frame update
    void Start()
    {
        transform.position = startingPosition;
        rBody = GetComponent<Rigidbody>();
    }

    // Update is called once per frame
    void Update()
    {
        if (Input.GetKeyDown("space") && speed < 100)
        {
            speed += 10;
        }
        rBody.velocity = new Vector3(0, 0, speed);
    }

    private void OnTriggerEnter(Collider other) 
    {
        spawnManager.SpawnTriggeredEntered();
    }
}
