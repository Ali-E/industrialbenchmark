/*
Copyright 2016 Siemens AG.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
package com.siemens.industialbenchmark.dynamics;


import static org.junit.Assert.assertEquals;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;
import java.util.Random;

import org.junit.Test;

import com.siemens.industrialbenchmark.datavector.action.ActionDelta;
import com.siemens.industrialbenchmark.datavector.state.MarkovianStateDescription;
import com.siemens.industrialbenchmark.datavector.state.ObservableState;
import com.siemens.industrialbenchmark.datavector.state.ObservableStateDescription;
import com.siemens.industrialbenchmark.dynamics.IndustrialBenchmarkDynamics;
import com.siemens.industrialbenchmark.externaldrivers.setpointgen.SetPointGenerator;
import com.siemens.industrialbenchmark.externaldrivers.setpointgen.SetPointGeneratorStateDescription;
import com.siemens.industrialbenchmark.properties.PropertiesException;
import com.siemens.industrialbenchmark.properties.PropertiesUtil;
import com.siemens.rl.interfaces.DataVector;
import com.siemens.rl.interfaces.ExternalDriver;

/**
 * This is a test class for the Dynamics.
 *
 * @author Michel Tokic
 */
public class TestDynamics {

	final int INIT_STEPS = 10000;
	final int MEM_STEPS = 10000;
	final long ACTION_SEED = 12345;

	/**
	 * This class tests that dynamics are repeatable,
	 * which is required by Particle-Swarm-Optimization.
	 *
	 *  1) 100000 steps are taken
	 *     (this initializes dynamics with a random trajectory)
	 *  2) observable and markov state are memorized
	 *  3) a random trajectory is performed and memorized
	 *  4) reset of states from (2) and call Environment.reset()
	 *  5) test if replay produces the same dynamics
	 *
	 * @throws IOException when there is an error reading the configuration file
	 * @throws PropertiesException if the configuration file is badly formatted
	 */
	@Test
	public void testRepeatibleDynamics() throws IOException, PropertiesException {


		// INSTANTIATE benchmark
		Properties props = PropertiesUtil.loadSetPointProperties(new File("src/main/resources/sim.properties"));
		SetPointGenerator lg = new SetPointGenerator(props);
		List<ExternalDriver> externalDrivers = new ArrayList<ExternalDriver>();
		externalDrivers.add(lg);
		IndustrialBenchmarkDynamics d = new IndustrialBenchmarkDynamics(props, externalDrivers);
		Random actionRand = new Random(System.currentTimeMillis());

        // 1) do 100000 random steps, in order to initialize dynamics
		final ActionDelta action = new ActionDelta(0.001f, 0.001f, 0.001f);
		for (int i=0; i<INIT_STEPS; i++) {
			action.setDeltaGain(2.f*(actionRand.nextFloat()-0.5f));
			action.setDeltaVelocity(2.f*(actionRand.nextFloat()-0.5f));
			action.setDeltaShift(2.f*(actionRand.nextFloat()-0.5f));
			d.step(action);
		}

		// 2) memorize current observable state and current markov state
		final ObservableState os = d.getState();
		final DataVector ms = d.getMarkovState();
		System.out.println("init o-state: " + os.toString());
		System.out.println("init m-state: " + ms.toString());


		// 3) perform test trajectory and memorize states
		actionRand.setSeed(ACTION_SEED);
		DataVector oStates[] = new DataVector[MEM_STEPS];
		DataVector mStates[] = new DataVector[MEM_STEPS];

		for (int i=0; i<MEM_STEPS; i++) {
			action.setDeltaGain(2.f*(actionRand.nextFloat()-0.5f));
			action.setDeltaVelocity(2.f*(actionRand.nextFloat()-0.5f));
			action.setDeltaShift(2.f*(actionRand.nextFloat()-0.5f));
			d.step(action);
			oStates[i] = d.getState();
			mStates[i] = d.getMarkovState();
		}

		// 4) reset dynamics & parameters and internal markov state
		d.reset();
		d.setMarkovState(ms);

		// 5) reperform test and check if values are consistent
		actionRand.setSeed(ACTION_SEED); // reproduce action sequence
		DataVector oState = null;
		DataVector mState = null;
		for (int i=0; i<MEM_STEPS; i++) {
			action.setDeltaGain(2.f*(actionRand.nextFloat()-0.5f));
			action.setDeltaVelocity(2.f*(actionRand.nextFloat()-0.5f));
			action.setDeltaShift(2.f*(actionRand.nextFloat()-0.5f));

			d.step(action);
			oState = d.getState();
			mState = d.getMarkovState();

			// check observable state
			assertEquals(oStates[i].getValue(ObservableStateDescription.SET_POINT), oState.getValue(ObservableStateDescription.SET_POINT), 0.0001);
			assertEquals(oStates[i].getValue(ObservableStateDescription.FATIGUE), oState.getValue(ObservableStateDescription.FATIGUE), 0.0001);
			assertEquals(oStates[i].getValue(ObservableStateDescription.CONSUMPTION), oState.getValue(ObservableStateDescription.CONSUMPTION), 0.0001);
			assertEquals(oStates[i].getValue(ObservableStateDescription.REWARD_TOTAL), oState.getValue(ObservableStateDescription.REWARD_TOTAL), 0.0001);

			//
			assertEquals(mStates[i].getValue(MarkovianStateDescription.CURRENT_OPERATIONAL_COST), mState.getValue(MarkovianStateDescription.CURRENT_OPERATIONAL_COST), 0.0001);
			assertEquals(mStates[i].getValue(MarkovianStateDescription.FATIGUE_LATENT_2), mState.getValue(MarkovianStateDescription.FATIGUE_LATENT_2), 0.0001);
			assertEquals(mStates[i].getValue(MarkovianStateDescription.FATIGUE_LATENT_1), mState.getValue(MarkovianStateDescription.FATIGUE_LATENT_1), 0.0001);

			assertEquals(mStates[i].getValue(MarkovianStateDescription.EFFECTIVE_ACTION_GAIN_BETA), mState.getValue(MarkovianStateDescription.EFFECTIVE_ACTION_GAIN_BETA), 0.0001);
			assertEquals(mStates[i].getValue(MarkovianStateDescription.EFFECTIVE_ACTION_VELOCITY_ALPHA), mState.getValue(MarkovianStateDescription.EFFECTIVE_ACTION_VELOCITY_ALPHA), 0.0001);
			assertEquals(mStates[i].getValue(MarkovianStateDescription.EFFECTIVE_SHIFT), mState.getValue(MarkovianStateDescription.EFFECTIVE_SHIFT), 0.0001);
			assertEquals(mStates[i].getValue(MarkovianStateDescription.MIS_CALIBRATION), mState.getValue(MarkovianStateDescription.MIS_CALIBRATION), 0.0001);

			assertEquals(mStates[i].getValue(SetPointGeneratorStateDescription.SET_POINT_CHANGE_RATE_PER_STEP), mState.getValue(SetPointGeneratorStateDescription.SET_POINT_CHANGE_RATE_PER_STEP), 0.0001);
			assertEquals(mStates[i].getValue(SetPointGeneratorStateDescription.SET_POINT_CURRENT_STEPS), mState.getValue(SetPointGeneratorStateDescription.SET_POINT_CURRENT_STEPS), 0.0001);
			assertEquals(mStates[i].getValue(SetPointGeneratorStateDescription.SET_POINT_LAST_SEQUENCE_STEPS), mState.getValue(SetPointGeneratorStateDescription.SET_POINT_LAST_SEQUENCE_STEPS), 0.0001);

			assertEquals(mStates[i].getValue(MarkovianStateDescription.REWARD_FATIGUE), mState.getValue(MarkovianStateDescription.REWARD_FATIGUE), 0.0001);
			assertEquals(mStates[i].getValue(MarkovianStateDescription.REWARD_CONSUMPTION), mState.getValue(MarkovianStateDescription.REWARD_CONSUMPTION), 0.0001);
		}

		System.out.println("last o-state 1st trajectory: " + oStates[oStates.length-1]);
		System.out.println("last o-state 2nd trajectory: " + oState);

		System.out.println("last m-state 1st trajectory: " + mStates[oStates.length-1]);
		System.out.println("last m-state 2nd trajectory: " + mState);
	}

	@Test
	public void testHistoryLength() throws IOException, PropertiesException {

		// INSTANTIATE benchmark
		Properties props = PropertiesUtil.loadSetPointProperties(new File("src/main/resources/sim.properties"));
		SetPointGenerator lg = new SetPointGenerator(props);
		List<ExternalDriver> externalDrivers = new ArrayList<ExternalDriver>();
		externalDrivers.add(lg);
		IndustrialBenchmarkDynamics d = new IndustrialBenchmarkDynamics(props, externalDrivers);

		int expHistSize = 0;
		for (String key : d.getMarkovState().getKeys()) {
			if (key.startsWith("OPERATIONALCOST_")) {
				expHistSize++;
			}
		}

		assertEquals(expHistSize, d.getOperationalCostsHistoryLength());
	}
}

