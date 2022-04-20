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
    public int currentSpeed = 0;
    public SpawnManager spawnManager;
    public bool increaseReady = true;

    void Start() 
    {
        rBody = GetComponent<Rigidbody>();
    }

    public override void OnEpisodeBegin()
    {

    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // Agent velocity
        sensor.AddObservation(rBody.velocity.z);
    }

    public override void OnActionReceived(ActionBuffers actions) 
    {
        int speed = actions.DiscreteActions[0];
        int zSpeed = 0;
        if (increaseReady)
        {
            increaseReady = false;
            currentSpeed = 2;
            // StartCoroutine(WaitAndIncrease());
            if (currentSpeed == 0) zSpeed = 0;
            if (currentSpeed == 1) zSpeed = 2;
            if (currentSpeed == 2) zSpeed = 4;
            if (currentSpeed == 3) zSpeed = 6;
            if (currentSpeed == 4) zSpeed = 8;
            if (currentSpeed == 5) zSpeed = 10;
            if (currentSpeed == 6) zSpeed = 12;
            if (currentSpeed == 7) zSpeed = 14;
            if (currentSpeed == 8) zSpeed = 16;
            if (currentSpeed == 9) zSpeed = 18;
            if (currentSpeed == 10) zSpeed = 20;
            rBody.velocity = new Vector3(0, 0, zSpeed);
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        if (Input.GetKey("0")) currentSpeed = 0;
        if (Input.GetKey("1")) currentSpeed = 1;
        if (Input.GetKey("2")) currentSpeed = 2;
        if (Input.GetKey("3")) currentSpeed = 3;
        if (Input.GetKey("4")) currentSpeed = 4;
        if (Input.GetKey("5")) currentSpeed = 5;
        if (Input.GetKey("6")) currentSpeed = 6;
        if (Input.GetKey("7")) currentSpeed = 7;
        if (Input.GetKey("8")) currentSpeed = 8;
        if (Input.GetKey("9")) currentSpeed = 9;
        ActionSegment<int> discreteActions = actionsOut.DiscreteActions;
        discreteActions[0] = currentSpeed;
    }

    private void OnTriggerEnter(Collider other) 
    {
        spawnManager.SpawnTriggeredEntered();
    }
    public IEnumerator WaitAndIncrease()
    {
        yield return new WaitForSecondsRealtime(20);
        if (currentSpeed < 10) 
        {
            currentSpeed += 1;
        }
        else currentSpeed = 0;
        increaseReady = true;
    }
}
