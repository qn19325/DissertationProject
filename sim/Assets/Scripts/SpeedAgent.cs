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
        if (speed == 0) zSpeed = 0;
        if (speed == 1) zSpeed = 10;
        if (speed == 2) zSpeed = 20;
        if (speed == 3) zSpeed = 30;
        if (speed == 4) zSpeed = 40;
        if (speed == 5) zSpeed = 50;
        if (speed == 6) zSpeed = 60;
        if (speed == 7) zSpeed = 70;
        if (speed == 8) zSpeed = 80;
        if (speed == 9) zSpeed = 90;
        rBody.velocity = new Vector3(0, 0, zSpeed);
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        // if (Input.GetKey("0")) currentSpeed = 0;
        // if (Input.GetKey("1")) currentSpeed = 1;
        // if (Input.GetKey("2")) currentSpeed = 2;
        // if (Input.GetKey("3")) currentSpeed = 3;
        // if (Input.GetKey("4")) currentSpeed = 4;
        // if (Input.GetKey("5")) currentSpeed = 5;
        // if (Input.GetKey("6")) currentSpeed = 6;
        // if (Input.GetKey("7")) currentSpeed = 7;
        // if (Input.GetKey("8")) currentSpeed = 8;
        // if (Input.GetKey("9")) currentSpeed = 9;
        if (increaseReady)
        {
            increaseReady = false;
            StartCoroutine(WaitAndIncrease());
        }
        ActionSegment<int> discreteActions = actionsOut.DiscreteActions;
        discreteActions[0] = currentSpeed;
    }

    private void OnTriggerEnter(Collider other) 
    {
        spawnManager.SpawnTriggeredEntered();
    }
    public IEnumerator WaitAndIncrease()
    {
        yield return new WaitForSecondsRealtime(5);
        currentSpeed += 1;
        increaseReady = true;

    }
}
