export function keepGUIAlive() {
  async function getRequest(url = '') {
    const response = await fetch(url, {
      method: 'GET',
      cache: 'no-cache',
    });
    return response;
  }

  const keepAliveServer = () => {
    const url = document.location;
    const route = '/flaskwebgui-keep-server-alive';
    getRequest(url + route).then((data) => {
      return data;
    });
  };

  if (!import.meta.env.NODE_ENV || import.meta.env.NODE_ENV === 'production') {
    document.addEventListener('DOMContentLoaded', () => {
      const intervalRequest = 3 * 1000;
      keepAliveServer();
      setInterval(keepAliveServer, intervalRequest);
    });
  }
}
