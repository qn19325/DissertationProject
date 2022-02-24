using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

public class SpeedAgent : Agent
{
    private Vector3 startingPosition = new Vector3(0, 0.51f, 0);
    Rigidbody rBody;
    public int speed = 0;
    public SpawnManager spawnManager;

    void Start() 
    {
        rBody = GetComponent<Rigidbody>();
    }

    public override void OnEpisodeBegin()
    {
        
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(rBody.velocity);
    }

    public override void OnActionReceived(ActionBuffers actions) 
    {
        int zSpeed = actions.DiscreteActions[0];
        rBody.velocity = new Vector3(0, 0, zSpeed);
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        if (Input.GetKey("0")) speed = 0;
        if (Input.GetKey("1")) speed = 10;
        if (Input.GetKey("2")) speed = 20;
        if (Input.GetKey("3")) speed = 30;
        if (Input.GetKey("4")) speed = 40;
        if (Input.GetKey("5")) speed = 50;
        if (Input.GetKey("6")) speed = 60;
        if (Input.GetKey("7")) speed = 70;
        if (Input.GetKey("8")) speed = 80;
        if (Input.GetKey("9")) speed = 90;
        ActionSegment<int> discreteActions = actionsOut.DiscreteActions;
        discreteActions[0] = speed;
    }

    private void OnTriggerEnter(Collider other) 
    {
        spawnManager.SpawnTriggeredEntered();
    }
}
