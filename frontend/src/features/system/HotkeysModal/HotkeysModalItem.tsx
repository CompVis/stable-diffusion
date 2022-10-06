import React from 'react';

interface HotkeysModalProps {
  hotkey: string;
  title: string;
  description?: string;
}

export default function HotkeysModalItem(props: HotkeysModalProps) {
  const { title, hotkey, description } = props;
  return (
    <div className="hotkey-modal-item">
      <div className="hotkey-info">
        <p className="hotkey-title">{title}</p>
        {description && <p className="hotkey-description">{description}</p>}
      </div>
      <div className="hotkey-key">{hotkey}</div>
    </div>
  );
}
