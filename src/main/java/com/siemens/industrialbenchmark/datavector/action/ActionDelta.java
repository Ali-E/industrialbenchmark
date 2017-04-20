/**
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
package com.siemens.industrialbenchmark.datavector.action;

import com.google.common.base.Preconditions;
import com.siemens.industrialbenchmark.datavector.DataVectorImpl;
import com.siemens.industrialbenchmark.properties.PropertiesException;

/**
 * This class keeps and checks the deltaA, deltaB and deltaC.
 *
 * @author Michel Tokic
 */
public class ActionDelta extends DataVectorImpl {

	private static final long serialVersionUID = 1159603096632053185L;

	protected static final double MAX_DELTA = 100.0; // HACK This was 10.0 before, but the default range is 100.0, so it produced runtime exceptions by default. why use this arbitrary value here anyway?

	/**
	 * Constructor with deltas and properties file.
	 * @param deltaVelocity The delta velocity to apply
	 * @param deltaGain The delta gain to apply
	 * @param deltaShift The delta shift to apply
	 * @throws PropertiesException
	 */
	public ActionDelta(final double deltaVelocity, final double deltaGain, final double deltaShift)
			throws PropertiesException
	{
		super(new ActionDeltaDescription());

		Preconditions.checkArgument(Math.abs(deltaVelocity) <= MAX_DELTA,
				"Math.abs(deltaA=%s) must be <= %s", deltaVelocity, MAX_DELTA);
		Preconditions.checkArgument(Math.abs(deltaGain) <= MAX_DELTA,
				"Math.abs(deltaB=%s) must be <= %s", deltaGain, MAX_DELTA);

		setValue(ActionDeltaDescription.DeltaVelocity, deltaVelocity);
		setValue(ActionDeltaDescription.DeltaGain, deltaGain);
		setValue(ActionDeltaDescription.DeltaShift, deltaShift);
	}

	/**
	 * @return the deltaA
	 */
	public double getDeltaVelocity() {
		return getValue(ActionDeltaDescription.DeltaVelocity);
	}

	/**
	 * @return the deltaB
	 */
	public double getDeltaGain() {
		return getValue(ActionDeltaDescription.DeltaGain);
	}

	/**
	 * @return the deltaC
	 */
	public double getDeltaShift() {
		return getValue(ActionDeltaDescription.DeltaShift);
	}

	/**
	 * @param deltaVelocity the delta velocity to set
	 */
	public void setDeltaVelocity(final double deltaVelocity) {
		Preconditions.checkArgument(Math.abs(deltaVelocity) <= MAX_DELTA,
				"Math.abs(deltaVelocity=%s) must be <= %s", deltaVelocity, MAX_DELTA);
		this.setValue(ActionDeltaDescription.DeltaVelocity, deltaVelocity);
	}

	/**
	 * @param deltaGain the delta gain to set
	 */
	public void setDeltaGain(final double deltaGain) {
		Preconditions.checkArgument(Math.abs(deltaGain) <= MAX_DELTA,
				"Math.abs(deltaGain=%s) must be <= %s", deltaGain, MAX_DELTA);
		this.setValue(ActionDeltaDescription.DeltaGain, deltaGain);
	}

	/**
	 * @param deltaShift The delta shift to set.
	 */
	public void setDeltaShift(final double deltaShift) {
		this.setValue(ActionDeltaDescription.DeltaShift, deltaShift);
	}
}

