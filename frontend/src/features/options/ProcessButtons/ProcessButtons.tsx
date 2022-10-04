import InvokeButton from './InvokeButton';
import CancelButton from './CancelButton';

/**
 * Buttons to start and cancel image generation.
 */
const ProcessButtons = () => {
  return (
    <div className="process-buttons">
      <InvokeButton />
      <CancelButton />
    </div>
  );
};

export default ProcessButtons;
