import time
from mmcv.fileio import FileClient, BaseStorageBackend


@FileClient.register_backend('MyMINIO')
class MINIOBackend(BaseStorageBackend):
    def __init__(self, bucket, path_mapping=None, scope=None, proxy=None,
                 **minio_cfg):
        self.bucket = bucket
        http_client = self.http_client(proxy)
        try:
            import minio
        except ImportError:
            raise ImportError('Please install minio to enable MINIOBackend.')
        self._client = minio.Minio(**minio_cfg, http_client=http_client)
        assert isinstance(path_mapping, dict) or path_mapping is None
        self.path_mapping = path_mapping

    @staticmethod
    def http_client(proxy=None):
        import urllib3
        import certifi
        import os
        if proxy is None:
            proxy = os.environ.get('MINIO_PROXY', None)
        if proxy is None:
            return None
        if proxy.startswith('socks'):
            try:
                from urllib3.contrib.socks import SOCKSProxyManager
            except ImportError:
                raise ImportError(
                    'Please install urllib3[socks] to enable socks proxy.')
            proxy_manager = SOCKSProxyManager
        elif proxy.startswith('http'):
            proxy_manager = urllib3.ProxyManager
        else:
            raise ValueError('Unknown proxy type!')
        return proxy_manager(
            proxy_url=proxy,
            timeout=urllib3.util.Timeout(connect=300, read=300),
            maxsize=10,
            cert_reqs='CERT_REQUIRED',
            ca_certs=os.environ.get('SSL_CERT_FILE') or certifi.where(),
            retries=urllib3.Retry(
                total=5,
                backoff_factor=0.2,
                status_forcelist=[500, 502, 503, 504]))

    def get(self, object_name):
        if self.path_mapping is not None:
            for k, v in self.path_mapping.items():
                object_name = object_name.replace(k, v)
        response = None
        while True:
            try:
                # If the loop continues, the server cannot be accessed
                response = self._client.get_object(self.bucket, object_name)
                ret = response.read()
            except Exception as e:
                print(e)
                time.sleep(0.1)
                continue
            finally:
                if response is not None:
                    response.close()
                    response.release_conn()
            break
        return ret

    def get_text(self, filepath, encoding=None):
        raise NotImplementedError

    def list_dir_or_file(self, dir_path, list_dir, list_file, suffix,
                         recursive):
        if not dir_path.endswith('/'):
            dir_path = dir_path + '/'
        list_objects = self._client.list_objects(self.bucket, prefix=dir_path,
                                                 recursive=recursive)
        return map(lambda o: o.object_name, list_objects)

    def exists(self, path):
        try:
            self._client.stat_object(self.bucket, path)
            return True
        except:
            return False